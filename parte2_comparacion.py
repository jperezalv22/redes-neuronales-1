"""
Parte 2: comparacion de colonia de hormigas y genetico, con y sin 2-opt,
en varias corridas independientes (semillas semilla, semilla+1, ...).

Sirve para saber si los metodos llegan siempre al mismo recorrido (lo que nos
da confianza de que es el optimo, o muy cerca) y que tanto aporta 2-opt.

Salidas en resultados/parte2/: comparacion.csv y comparacion.md

Ejemplo:
    python parte2_comparacion.py                 # 10 corridas, ~2 min
    python parte2_comparacion.py --corridas 3    # rapido
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from tsp import DatosTSP, algoritmo_genetico_tsp, cargar_config, colonia_hormigas


def leer_argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configuracion_tsp.json")
    p.add_argument("--corridas", type=int, default=10)
    p.add_argument("--semilla", type=int, default=2026)
    p.add_argument("--valor-hora", type=float)
    p.add_argument("--salida", default="resultados/parte2")
    return p.parse_args()


def main():
    a = leer_argumentos()
    cfg = cargar_config(a.config)
    h = cfg["valor_hora_eur"] if a.valor_hora is None else a.valor_hora
    datos = DatosTSP(cfg["consumo_l_100km"], cfg["precio_combustible_eur_l"])
    C = datos.costo(h)

    # mismos parametros por defecto que parte2_tsp.py
    configuraciones = {
        "ACO + 2-opt": lambda rng: colonia_hormigas(C, rng, iteraciones=150),
        "ACO sin 2-opt": lambda rng: colonia_hormigas(C, rng, iteraciones=150, usar_dos_opt=False),
        "GA + 2-opt": lambda rng: algoritmo_genetico_tsp(C, rng, generaciones=300),
        "GA sin 2-opt": lambda rng: algoritmo_genetico_tsp(C, rng, generaciones=300, usar_dos_opt=False),
    }

    costos = {}
    tiempos = {}
    for nombre, correr in configuraciones.items():
        cs, ts = [], []
        for k in range(a.corridas):
            t0 = time.time()
            res = correr(np.random.default_rng(a.semilla + k))
            ts.append(time.time() - t0)
            cs.append(res["costo"])
        costos[nombre], tiempos[nombre] = np.array(cs), np.mean(ts)
        print(f"{nombre:15s} media {np.mean(cs):,.2f}  mejor {np.min(cs):,.2f}  peor {np.max(cs):,.2f}")

    mejor_conocido = min(c.min() for c in costos.values())
    filas = []
    for nombre, cs in costos.items():
        filas.append({
            "configuracion": nombre,
            "media": round(cs.mean(), 2), "desv": round(cs.std(ddof=1), 2),
            "mejor": round(cs.min(), 2), "peor": round(cs.max(), 2),
            "brecha_media_%": round(100 * (cs.mean() / mejor_conocido - 1), 2),
            "veces_mejor_conocido": int(np.sum(cs < mejor_conocido + 0.01)),
            "segundos_por_corrida": round(tiempos[nombre], 1),
        })

    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    with open(salida / "comparacion.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(filas)

    lineas = [f"Valor de la hora: {h:g} €/h. Corridas: {a.corridas}. Mejor costo conocido: {mejor_conocido:,.2f} €\n",
              "| Configuración | Media (€) | Desv. (€) | Mejor (€) | Peor (€) | Brecha media | Veces que llega al mejor | s/corrida |",
              "|---|---|---|---|---|---|---|---|"]
    for r in filas:
        lineas.append(f"| {r['configuracion']} | {r['media']:,.2f} | {r['desv']:,.2f} | {r['mejor']:,.2f} | "
                      f"{r['peor']:,.2f} | {r['brecha_media_%']:.2f} % | {r['veces_mejor_conocido']}/{a.corridas} | "
                      f"{r['segundos_por_corrida']} |")
    (salida / "comparacion.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
