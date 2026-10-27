"""
gen_datos.py — Planta virtual de la barra térmica (generador del dataset de referencia)
=======================================================================================
Simula por diferencias finitas una barra de aluminio de 30 cm calentada en un extremo
con una resistencia de ~1,5 W durante 5 min y enfriada 10 min, con pérdidas al ambiente.
Genera los CSV con el mismo formato que publica el ESP32 (tiempo + 6 sensores).

    python gen_datos.py                 -> data/ensayo_base.csv y data/ensayo_deriva.csv

El sensor S6 (x = 12 cm) es el "sensor oculto": NO se usa para entrenar, solo para validar
el sensor virtual. En el montaje real corresponde a un sexto DS18B20 que el profesor
retira del topic público.
"""
import numpy as np, pandas as pd, os, argparse

# ---------------- parámetros físicos (aluminio 6060, barra Ø10 mm) ----------------
L      = 0.30          # m   longitud de la barra
D      = 0.010         # m   diámetro
ALPHA  = 9.7e-5        # m²/s difusividad térmica del aluminio
RHO_C  = 2.43e6        # J/(m³·K)  rho*c
P_HEAT = 2.5           # W   potencia de la resistencia
L_HEAT = 0.02          # m   longitud calentada (extremo x=0)
T_ON   = 300.0         # s   duración del pulso de calor
T_END  = 900.0         # s   duración total del ensayo
W_X    = 0.006         # m   anchura de transición de la zona calentada
W_T    = 10.0          # s   constante de tiempo del encendido/apagado (inercia de la resistencia)
T_AMB  = 22.0          # °C
H_CONV = {"base": 8.0, "deriva": 25.0}   # W/(m²K): convección natural / con ventilador
X_SENS = [0.03, 0.09, 0.15, 0.21, 0.27, 0.01]   # S1..S5 visibles, S6 oculto (1 cm, junto al calefactor)
DT_MUESTREO = 2.0      # s   periodo de publicación del ESP32
SIGMA_RUIDO = 0.25     # °C  ruido del DS18B20 + contacto

def fuente_x(x):
    """Perfil espacial suave de la fuente (K/s por unidad de q0)."""
    return 1.0 / (1.0 + np.exp((x - L_HEAT) / W_X))

def fuente_t(t):
    return 1.0 / (1.0 + np.exp((t - T_ON) / W_T))

def simular(h_conv, seed=0, n=151):
    A = np.pi * (D / 2) ** 2
    Pm = np.pi * D
    h_eff = h_conv * Pm / (RHO_C * A)        # 1/s  pérdidas por unidad de capacidad térmica
    q0 = P_HEAT / (RHO_C * A * L_HEAT)       # K/s  en la zona calentada
    x = np.linspace(0, L, n); dx = x[1] - x[0]
    dt = 0.4 * dx ** 2 / ALPHA
    nt = int(np.ceil(T_END / dt)); dt = T_END / nt
    T = np.full(n, T_AMB)
    fx = fuente_x(x)
    t_out = np.arange(0, T_END + 1e-9, DT_MUESTREO)
    out = np.zeros((len(t_out), n)); k = 0
    for i in range(nt + 1):
        t = i * dt
        if k < len(t_out) and t >= t_out[k] - 1e-9:
            out[k] = T; k += 1
        lap = np.zeros(n)
        lap[1:-1] = (T[2:] - 2 * T[1:-1] + T[:-2]) / dx ** 2
        lap[0] = 2 * (T[1] - T[0]) / dx ** 2          # extremos aislados (Neumann)
        lap[-1] = 2 * (T[-2] - T[-1]) / dx ** 2
        T = T + dt * (ALPHA * lap - h_eff * (T - T_AMB) + q0 * fx * fuente_t(t))
    return x, t_out, out, h_eff, q0

def a_csv(x, t, campo, path, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"t_s": t})
    for i, xs in enumerate(X_SENS, start=1):
        j = int(round(xs / (x[1] - x[0])))
        df[f"S{i}"] = np.round(campo[:, j] + rng.normal(0, SIGMA_RUIDO, len(t)), 2)
    df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = []
    for nombre, seed in (("base", 1), ("deriva", 2)):
        x, t, campo, h_eff, q0 = simular(H_CONV[nombre], seed)
        df = a_csv(x, t, campo, f"{a.out}/ensayo_{nombre}.csv", seed)
        np.save(f"{a.out}/campo_{nombre}.npy", campo)     # verdad completa T(x,t), solo para el profesor
        meta.append((nombre, H_CONV[nombre], h_eff, q0, campo.max()))
        print(f"{nombre}: h_conv={H_CONV[nombre]} W/m²K -> h_eff={h_eff:.2e} 1/s, q0={q0:.3f} K/s, Tmax={campo.max():.1f} °C")
    np.save(f"{a.out}/x_malla.npy", x)
    with open(f"{a.out}/README_datos.txt", "w") as f:
        f.write("Ensayos de la barra de aluminio (planta virtual). Columnas: t_s, S1..S5 (x = 3, 9, 15, 21, 27 cm), S6 oculto (x = 1 cm).\n")
        f.write(f"L={L} m, D={D} m, alpha={ALPHA} m2/s, P={P_HEAT} W en los primeros {L_HEAT*100:.0f} cm, pulso {T_ON:.0f} s, T_amb={T_AMB} C, muestreo {DT_MUESTREO} s, ruido {SIGMA_RUIDO} C.\n")
        for m in meta: f.write(f"{m[0]}: h_conv={m[1]} -> h_eff={m[2]:.3e} 1/s, q0={m[3]:.3f} K/s, Tmax={m[4]:.1f}\n")
