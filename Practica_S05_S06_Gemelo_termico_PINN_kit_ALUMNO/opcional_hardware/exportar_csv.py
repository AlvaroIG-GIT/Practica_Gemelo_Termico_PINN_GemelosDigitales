"""Exporta un ensayo desde InfluxDB al formato CSV de la práctica (t_s, S1..S5).
    python exportar_csv.py --pareja p01 --out data/mi_ensayo.csv
"""
import argparse, pandas as pd, requests
ap = argparse.ArgumentParser(); ap.add_argument("--pareja", required=True); ap.add_argument("--host", default="localhost")
ap.add_argument("--out", required=True); a = ap.parse_args()
q = f"SELECT T, t_s FROM barra WHERE pareja='{a.pareja}' AND t_s >= 0 GROUP BY sensor"
r = requests.get(f"http://{a.host}:8086/query", params={"db": "lab", "q": q}).json()
frames = {}
for s in r["results"][0]["series"]:
    df = pd.DataFrame(s["values"], columns=s["columns"]); frames[s["tags"]["sensor"]] = df.set_index("t_s")["T"]
out = pd.DataFrame(frames).sort_index().reset_index().rename(columns={"index": "t_s"})
out = out[out.t_s <= 900].round(2); out.to_csv(a.out, index=False); print(out.shape, "->", a.out)
