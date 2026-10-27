"""
pinn_barra.py — PINN de la ecuación del calor 1D para el gemelo térmico de la barra
====================================================================================
Modelo físico (dimensional):
    dT/dt = alpha * d²T/dx² - h * (T - T_amb) + q(x, t)
Adimensional (x* = x/L, t* = t/T_REF, u = (T - T_amb)/DT_REF):
    du/dt* = a * d²u/dx*² - b * u + s(x*, t*)      con  a = alpha*T_REF/L²,  b = h*T_REF
La red aprende u(x*, t*); en el problema inverso a y b son parámetros entrenables.

Uso mínimo (ver cuaderno):
    from pinn_barra import *
    datos = cargar_ensayo("data/ensayo_base.csv")
    pinn  = PINN(inverso=False, a=..., b=...)
    hist  = entrenar(pinn, datos, n_adam=4000, n_lbfgs=300, lam_pde=1.0)
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, time, math

torch.set_default_dtype(torch.float32)
DEV = torch.device("cpu")

# --------------------- constantes del montaje (deben coincidir con el ESP32) ---------------------
L = 0.30; T_REF = 900.0; L_HEAT = 0.02; T_ON = 300.0; W_X = 0.006; W_T = 10.0
X_SENS = {"S1": 0.03, "S2": 0.09, "S3": 0.15, "S4": 0.21, "S5": 0.27, "S6": 0.01}
SENSORES_VISIBLES = ["S1", "S2", "S3", "S4", "S5"]
SENSOR_OCULTO = "S6"

# ------------------------------------- datos -------------------------------------------------
class Ensayo:
    """Contiene el CSV ya adimensionalizado. DT_REF se fija con el máximo observado."""
    def __init__(self, df, sensores=SENSORES_VISIBLES, t_max=None, como=None):
        """t_max: usar solo los datos hasta ese instante [s]. como: otro Ensayo cuya escala (T_amb, DT_REF) se reutiliza."""
        if t_max is not None: df = df[df.t_s <= t_max]
        self.df = df.reset_index(drop=True); self.sensores = list(sensores)
        if como is not None:
            self.T_amb, self.DT_REF = como.T_amb, como.DT_REF
        else:
            self.T_amb = float(df[sensores].iloc[:5].values.mean())      # ambiente = primeras lecturas
            self.DT_REF = float(df[sensores].values.max() - self.T_amb)
        xs, ts, us = [], [], []
        for s in sensores:
            xs.append(np.full(len(df), X_SENS[s] / L)); ts.append(df.t_s.values / T_REF)
            us.append((df[s].values - self.T_amb) / self.DT_REF)
        self.x = torch.tensor(np.concatenate(xs), dtype=torch.float32)[:, None]
        self.t = torch.tensor(np.concatenate(ts), dtype=torch.float32)[:, None]
        self.u = torch.tensor(np.concatenate(us), dtype=torch.float32)[:, None]
    def a_temperatura(self, u):  return u * self.DT_REF + self.T_amb
    def serie(self, sensor):     return self.df.t_s.values, self.df[sensor].values

def cargar_ensayo(path, **kw):
    """cargar_ensayo("data/p07_base.csv")            -> ensayo completo
       cargar_ensayo(..., t_max=300, como=datos)     -> solo hasta 300 s, con la escala de `datos`
       cargar_ensayo(..., sensores=["S1","S3","S5"]) -> solo esos sensores"""
    return Ensayo(pd.read_csv(path), **kw)

def amplitud_fuente(datos, P_heat=2.5, D=0.010, rho_c=2.43e6):
    """Amplitud adimensional s0 de la fuente a partir de la potencia de la resistencia y la geometría."""
    A = np.pi * (D / 2) ** 2
    q0 = P_heat / (rho_c * A * L_HEAT)            # K/s en la zona calentada
    return q0 * T_REF / datos.DT_REF

def valores_de_tablas(D=0.010, rho_c=2.43e6, alpha=9.7e-5, h_conv=8.0):
    """(a, b) adimensionales para el aluminio 'de manual' con convección natural."""
    A = np.pi * (D / 2) ** 2
    return adimensional(alpha, h_conv * np.pi * D / (rho_c * A))

def dimensional(a, b):
    """(a, b) adimensionales -> (alpha [m²/s], h [1/s])"""
    return a * L**2 / T_REF, b / T_REF

def adimensional(alpha, h):
    return alpha * T_REF / L**2, h * T_REF

# ---------------------------------- fuente conocida -------------------------------------------
def fuente(x, t):
    """Perfil s(x*,t*) sin la amplitud: sigmoide en x (zona calentada) y en t (pulso)."""
    fx = torch.sigmoid(-(x * L - L_HEAT) / W_X)
    ft = torch.sigmoid(-(t * T_REF - T_ON) / W_T)
    return fx * ft

# ------------------------------------- red ----------------------------------------------------
class Fourier(nn.Module):
    """Rasgos de Fourier aleatorios en la entrada: mitigan el sesgo espectral (la tendencia de un MLP a
    aprender antes las componentes suaves). En este problema la solución es suave y su efecto es pequeño;
    compruébalo con fourier=False en la exploración 3 del paso 2."""
    def __init__(self, n=32, sigma=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(2, n, generator=g) * sigma)
    def forward(self, xt):
        z = 2 * math.pi * xt @ self.B
        return torch.cat([torch.sin(z), torch.cos(z)], dim=1)

class PINN(nn.Module):
    def __init__(self, inverso=False, a=1.0, b=1.0, s0=None, ancho=64, capas=3, fourier=True, seed=0, fuente_libre=False):
        super().__init__()
        torch.manual_seed(seed)
        self.ff = Fourier(seed=seed) if fourier else None
        n_in = 64 if fourier else 2
        mods, n = [], n_in
        for _ in range(capas): mods += [nn.Linear(n, ancho), nn.Tanh()]; n = ancho
        mods += [nn.Linear(n, 1)]
        self.red = nn.Sequential(*mods)
        # parámetros físicos en escala logarítmica (siempre positivos)
        self.log_a = nn.Parameter(torch.tensor(math.log(a)), requires_grad=inverso)
        self.log_b = nn.Parameter(torch.tensor(math.log(b)), requires_grad=inverso)
        self.log_s0 = nn.Parameter(torch.tensor(math.log(s0)), requires_grad=fuente_libre) if s0 else None   # fuente_libre: potencia desconocida (súper reto b)
        self.inverso = inverso
    @property
    def a(self): return self.log_a.exp()
    @property
    def b(self): return self.log_b.exp()
    def parametros(self):
        """(alpha [m²/s], h [1/s]) identificados, como floats."""
        return dimensional(self.log_a.detach().exp().item(), self.log_b.detach().exp().item())
    def amplitud_identificada(self):
        """s0 identificado (súper reto b): potencia = s0 * DT_REF / T_REF * rho_c * A * L_HEAT."""
        return self.log_s0.detach().exp().item()
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        z = self.ff(xt) if self.ff is not None else xt
        return t * self.red(z)                     # restricción dura: u(x, 0) = 0

    def residuo(self, x, t):
        x = x.requires_grad_(True); t = t.requires_grad_(True)
        u = self(x, t)
        g = lambda y, v: torch.autograd.grad(y, v, torch.ones_like(y), create_graph=True)[0]
        u_t = g(u, t); u_x = g(u, x); u_xx = g(u_x, x)
        if self.log_s0 is not None:
            s0 = self.log_s0.exp(); s = s0 * fuente(x, t)
        else:
            s0, s = 1.0, 0.0
        # el residuo se divide por la amplitud de la fuente: así L_pde es un error RELATIVO y no lo
        # domina la zona del calefactor (sin este escalado la PINN no converge; pruébalo)
        return (u_t - self.a * u_xx + self.b * u - s) / s0, u_x

# ---------------------------------- entrenamiento ---------------------------------------------
def puntos_colocacion(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g); t = torch.rand(n, 1, generator=g)
    # refuerzo cerca del calefactor y del apagado, donde la solución cambia más deprisa
    x[: n // 5] = torch.rand(n // 5, 1, generator=g) * 0.15
    t[n // 5 : 2 * n // 5] = T_ON / T_REF + (torch.rand(n // 5, 1, generator=g) - 0.5) * 0.1
    return x, t

def entrenar(pinn, datos, n_adam=4000, n_lbfgs=300, lam_pde=1.0, lam_bc=1.0, lam_dat=1.0,
             n_col=4000, lr=2e-3, congelar_frac=0.1, verbose=True, cada=500):
    """Adam (con los parámetros físicos congelados el primer 10 %) y después L-BFGS."""
    xc, tc = puntos_colocacion(n_col)
    tb = torch.rand(400, 1); xb = torch.cat([torch.zeros(200, 1), torch.ones(200, 1)])
    hist = {"it": [], "L": [], "L_dat": [], "L_pde": [], "L_bc": [], "a": [], "b": []}
    fis = [pinn.log_a, pinn.log_b] + ([pinn.log_s0] if pinn.log_s0 is not None and pinn.log_s0.requires_grad else [])
    red = [p for n, p in pinn.named_parameters() if not n.startswith("log_")]
    opt = torch.optim.Adam([{"params": red, "lr": lr}, {"params": fis, "lr": lr * 2}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_adam, eta_min=lr * 0.05)

    def perdidas():
        u = pinn(datos.x, datos.t)
        L_dat = ((u - datos.u) ** 2).mean()
        r, _ = pinn.residuo(xc, tc); L_pde = (r ** 2).mean()
        _, ux = pinn.residuo(xb, tb); L_bc = (ux ** 2).mean()
        return lam_dat * L_dat + lam_pde * L_pde + lam_bc * L_bc, L_dat, L_pde, L_bc

    t0 = time.time()
    for it in range(n_adam):
        activo = pinn.inverso and it >= congelar_frac * n_adam
        for p in fis: p.requires_grad_(activo)
        opt.zero_grad(); Ltot, L_dat, L_pde, L_bc = perdidas(); Ltot.backward(); opt.step(); sched.step()
        if it % cada == 0 or it == n_adam - 1:
            for k, v in zip(hist, [it, Ltot, L_dat, L_pde, L_bc, pinn.a, pinn.b]): hist[k].append(float(v.detach()) if torch.is_tensor(v) else float(v))
            if verbose:
                print(f"Adam {it:5d}  L={Ltot:.2e}  dat={L_dat:.2e}  pde={L_pde:.2e}  bc={L_bc:.2e}"
                      f"  a={pinn.a:.3f} b={pinn.b:.3f}  ({time.time()-t0:.0f}s)")
    if n_lbfgs > 0:
        for p in fis: p.requires_grad_(pinn.inverso)
        params = red + (fis if pinn.inverso else [])
        opt2 = torch.optim.LBFGS(params, max_iter=n_lbfgs, history_size=50, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-9, tolerance_change=1e-12)
        def closure():
            opt2.zero_grad(); Ltot, *_ = perdidas(); Ltot.backward(); return Ltot
        opt2.step(closure)
        Ltot, L_dat, L_pde, L_bc = perdidas()
        for k, v in zip(hist, [n_adam + n_lbfgs, Ltot, L_dat, L_pde, L_bc, pinn.a, pinn.b]): hist[k].append(float(v.detach()) if torch.is_tensor(v) else float(v))
        if verbose: print(f"L-BFGS      L={Ltot:.2e}  dat={L_dat:.2e}  pde={L_pde:.2e}  a={pinn.a:.3f} b={pinn.b:.3f}  ({time.time()-t0:.0f}s)")
    return hist

# --------------------------------- utilidades de evaluación -----------------------------------
@torch.no_grad()
def predecir(pinn, datos, x_m, t_s):
    """Temperatura predicha en x [m] (array) y t [s] (array) -> matriz (len(t), len(x))."""
    X, T = np.meshgrid(np.asarray(x_m) / L, np.asarray(t_s) / T_REF)
    u = pinn(torch.tensor(X.reshape(-1, 1), dtype=torch.float32), torch.tensor(T.reshape(-1, 1), dtype=torch.float32))
    return datos.a_temperatura(u.numpy().reshape(X.shape))

def error_sensor(pinn, datos, sensor=SENSOR_OCULTO):
    t, T_med = datos.serie(sensor)
    T_pred = predecir(pinn, datos, [X_SENS[sensor]], t)[:, 0]
    return float(np.sqrt(np.mean((T_pred - T_med) ** 2))), T_pred

def error_por_ventanas(modelo, datos, ancho_s=60.0, sensores=SENSORES_VISIBLES):
    """RMSE [°C] entre lo que predice el gemelo y lo que miden los sensores, por ventana temporal.
    Es la señal de deriva: si sube de forma sostenida, la planta ya no es la que el gemelo aprendió."""
    t_edges = np.arange(0, T_REF + 1e-9, ancho_s); centros, rmse = [], []
    for t0 in t_edges[:-1]:
        m = (datos.df.t_s >= t0) & (datos.df.t_s < t0 + ancho_s)
        if m.sum() == 0: continue
        err = []
        for s in sensores:
            t, T_med = datos.serie(s); T_pred = predecir(modelo, datos, [X_SENS[s]], t[m.values])[:, 0]
            err.append((T_pred - T_med[m.values]) ** 2)
        centros.append(t0 + ancho_s / 2); rmse.append(float(np.sqrt(np.mean(np.concatenate(err)))))
    return np.array(centros), np.array(rmse)

def guardar(pinn, path):  torch.save(pinn.state_dict(), path)
def cargar(pinn, path):   pinn.load_state_dict(torch.load(path)); return pinn

class MLP(nn.Module):
    """Red convencional (sin física) para el paso 1 del cuaderno."""
    def __init__(self, ancho=64, capas=3, seed=0):
        super().__init__(); torch.manual_seed(seed)
        mods, n = [], 2
        for _ in range(capas): mods += [nn.Linear(n, ancho), nn.Tanh()]; n = ancho
        self.red = nn.Sequential(*mods, nn.Linear(n, 1))
    def forward(self, x, t): return self.red(torch.cat([x, t], 1))

def entrenar_mlp(mlp, datos, n=3000, lr=2e-3):
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    for it in range(n):
        opt.zero_grad(); L = ((mlp(datos.x, datos.t) - datos.u) ** 2).mean(); L.backward(); opt.step()
    return float(L.detach())
