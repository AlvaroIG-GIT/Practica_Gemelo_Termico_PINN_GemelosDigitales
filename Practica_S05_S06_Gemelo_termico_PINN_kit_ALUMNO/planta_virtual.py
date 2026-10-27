"""
planta_virtual.py — «Vuestra» barra de aluminio, simulada en el PC
===================================================================
Cada pareja tiene una barra distinta (misma geometría, propiedades ligeramente diferentes,
deterministas a partir del identificador de la pareja). El programa realiza un ensayo de 15 min
(5 min de calentamiento + 10 min de enfriamiento) y guarda el CSV con el mismo formato que
publicaría el ESP32 del montaje real.

    python planta_virtual.py --pareja p07                      -> data/p07_base.csv
    python planta_virtual.py --pareja p07 --ensayo ventilador  -> data/p07_ventilador.csv   (reto)
    python planta_virtual.py --pareja p07 --ensayo aislante    -> data/p07_aislante.csv     (reto, alternativa)
    python planta_virtual.py --pareja p07 --misterio           -> data/p07_misterio.csv     (súper reto: otro metal)

Columnas del CSV: t_s, S1..S5 (x = 3, 9, 15, 21, 27 cm) y S6, el sensor OCULTO en x = 1 cm,
que solo sirve para comprobar el sensor virtual (no se usa para entrenar).
"""
import argparse, os, sys, time, hashlib
import numpy as np, pandas as pd
import gen_datos as G

METALES = {"cobre": 1.11e-4, "laton": 3.4e-5, "bronce": 1.9e-5, "acero": 1.2e-5}
ALPHA_ALUMINIO = 9.7e-5        # valor de referencia; NO se lee de gen_datos, que se modifica al simular

def semilla(pareja):
    return int(hashlib.sha256(pareja.strip().lower().encode()).hexdigest()[:8], 16)

def parametros_pareja(pareja):
    """(alpha, h_conv, metal_del_misterio) de esa pareja. Deterministas y estables entre llamadas."""
    rng = np.random.default_rng(semilla(pareja))
    alpha = ALPHA_ALUMINIO * (1 + rng.uniform(-0.12, 0.12))  # aleación / contacto: ±12 %
    h_conv = 8.0 * (1 + rng.uniform(-0.25, 0.25))            # convección natural: ±25 %
    metal = list(METALES)[rng.integers(len(METALES))]
    return alpha, h_conv, metal

def ensayo(pareja, tipo="base", misterio=False, out="data", silencioso=False):
    alpha, h_conv, metal = parametros_pareja(pareja)
    factor = {"base": 1.0, "ventilador": 3.0, "aislante": 0.4}[tipo]
    if misterio: alpha = METALES[metal]
    nombre = f"{pareja}_{'misterio' if misterio else tipo}"
    if not silencioso:
        print(f"Barra de la pareja {pareja} · ensayo «{'misterio' if misterio else tipo}»")
        print("Encendiendo la resistencia (2,5 W) ... ", end="", flush=True)
        for k in range(15):
            time.sleep(0.15); print("■" if k < 5 else "□", end="", flush=True)
        print("  15 min simulados")
    alpha_previo = G.ALPHA
    try:                                   # G.ALPHA es global: se restaura siempre, para que dos
        G.ALPHA = alpha                    # ensayos seguidos de la misma pareja usen la MISMA barra
        x, t, campo, h_eff, q0 = G.simular(h_conv * factor, seed=semilla(pareja) % 1000)
    finally:
        G.ALPHA = alpha_previo
    os.makedirs(out, exist_ok=True)
    path = f"{out}/{nombre}.csv"
    G.a_csv(x, t, campo, path, seed=(semilla(pareja) + semilla(tipo)) % 9973)
    if not silencioso:
        print(f"Ensayo guardado en {path}  (T máxima leída: {campo.max():.1f} °C)")
    return path

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pareja", help="identificador de la pareja, p. ej. p07")
    ap.add_argument("--ensayo", default="base", choices=["base", "ventilador", "aislante"])
    ap.add_argument("--misterio", action="store_true", help="barra de otro metal (súper reto)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--clave", action="store_true", help="(profesor) valores reales de la barra de --pareja")
    ap.add_argument("--claves", type=int, metavar="N", help="(profesor) tabla de claves de p01 a pN")
    a = ap.parse_args()
    A = np.pi * (G.D / 2) ** 2
    def fila(pareja):
        al, hc, metal = parametros_pareja(pareja)
        h_eff = hc * np.pi * G.D / (G.RHO_C * A)
        return (f"{pareja:>5}  {al:.3e}  {hc:5.2f}   {h_eff:.3e}   {3*h_eff:.3e}   "
                f"{0.4*h_eff:.3e}   {metal:<7} {METALES[metal]:.2e}")
    if a.claves:
        print("pareja     alpha     h_conv     h base      h ventilador  h aislante   misterio")
        for i in range(1, a.claves + 1): print(fila(f"p{i:02d}"))
        sys.exit()
    if a.clave:
        print("pareja     alpha     h_conv     h base      h ventilador  h aislante   misterio")
        print(fila(a.pareja)); sys.exit()
    if not a.pareja: ap.error("hace falta --pareja (o --claves N)")
    ensayo(a.pareja, a.ensayo, a.misterio, a.out)
