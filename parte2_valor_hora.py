"""
Estudio del valor de la hora del vendedor: ¿cambia la ruta optima?

Para cada valor de la hora h se resuelve el TSP con colonia de hormigas
(+2-opt) y se guarda el mejor recorrido.

Truco para que el ruido del algoritmo no nos engañe: al final, para cada h
se evaluan TODOS los recorridos encontrados (con cualquier h) usando el costo
de ese h, y nos quedamos con el mas barato. Asi, si para h = 20 el algoritmo
tuvo mala suerte pero la ruta encontrada con h = 30 es mejor, la usamos.

Como el costo es  C_h = h * tiempo + (peajes + combustible),  para un
recorrido fijo el costo total es una recta en h. El recorrido optimo es el
de la "envolvente inferior" de esas rectas, y cambia solo en los cruces.

Salidas en resultados/parte2/:
  - valor_hora.csv / valor_hora.md : km, horas, peajes y costo por cada h
  - valor_hora.png                 : como cambian km, horas y peajes con h
  - rutas_valor_hora.png           : ruta con el h mas bajo y el mas alto

Ejemplo:
    python parte2_valor_hora.py
    python parte2_valor_hora.py --valores 0 10 20 50 100 --iteraciones 60
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsp import DatosTSP, cargar_config, colonia_hormigas, costo_ruta
from tsp.mapa import dibujar_contorno, dibujar_ruta


def leer_argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configuracion_tsp.json")
    p.add_argument("--valores", nargs="+", type=float,
                   default=[0, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200])
    p.add_argument("--iteraciones", type=int, default=60)
    p.add_argument("--hormigas", type=int, default=20)
    p.add_argument("--semilla", type=int, default=2026)
    p.add_argument("--salida", default="resultados/parte2")
    return p.parse_args()


def aristas(ruta):
    """Conjunto de aristas (sin direccion) de un recorrido cerrado."""
    return {frozenset((ruta[k], ruta[(k + 1) % len(ruta)])) for k in range(len(ruta))}


def main():
    a = leer_argumentos()
    cfg = cargar_config(a.config)
    datos = DatosTSP(cfg["consumo_l_100km"], cfg["precio_combustible_eur_l"])
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    # 1) resolver para cada h
    encontradas = []
    for h in a.valores:
        rng = np.random.default_rng(a.semilla)
        res = colonia_hormigas(datos.costo(h), rng, n_hormigas=a.hormigas, iteraciones=a.iteraciones)
        encontradas.append(res["ruta"])
        print(f"h = {h:6.1f} EUR/h  costo = {res['costo']:,.2f} EUR")

    # 2) para cada h, la mejor de todas las rutas encontradas
    filas, rutas = [], []
    base = None
    for h in a.valores:
        C = datos.costo(h)
        ruta = min(encontradas, key=lambda r: costo_ruta(r, C))
        rutas.append(ruta)
        if base is None:
            base = aristas(ruta)
        d = datos.desglose(ruta, h)
        filas.append({
            "valor_hora": h,
            "km": round(d["km"], 1),
            "horas": round(d["horas"], 2),
            "peajes_eur": round(d["peajes_eur"], 2),
            "combustible_eur": round(d["combustible_eur"], 2),
            "costo_total_eur": round(d["costo_total_eur"], 2),
            "aristas_distintas_vs_primero": len(aristas(ruta) - base),
        })

    with open(salida / "valor_hora.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    lineas = ["| Valor hora (€/h) | km | horas | peajes (€) | combustible (€) | costo total (€) | aristas distintas |",
              "|---|---|---|---|---|---|---|"]
    for r in filas:
        lineas.append(f"| {r['valor_hora']:g} | {r['km']:,} | {r['horas']} | {r['peajes_eur']} | "
                      f"{r['combustible_eur']} | {r['costo_total_eur']:,} | {r['aristas_distintas_vs_primero']} |")
    (salida / "valor_hora.md").write_text("\n".join(lineas) + "\n")
    print("\n".join(lineas))

    cambios = [r["valor_hora"] for r in filas if r["aristas_distintas_vs_primero"] > 0]
    if cambios:
        print(f"\nLa ruta cambia respecto a h = {a.valores[0]:g} a partir de h = {cambios[0]:g} EUR/h")
    else:
        print("\nLa ruta no cambia en el rango estudiado")

    graficar(filas, salida)
    graficar_rutas(datos, rutas, a.valores, salida)


def graficar(filas, salida):
    h = [r["valor_hora"] for r in filas]
    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, clave, nombre in zip(ejes, ["km", "horas", "peajes_eur"],
                                 ["distancia total (km)", "tiempo total (h)", "peajes (€)"]):
        ax.plot(h, [r[clave] for r in filas], "o-")
        ax.set_xlabel("valor de la hora (€/h)")
        ax.set_ylabel(nombre)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(salida / "valor_hora.png", dpi=120)
    plt.close(fig)


def graficar_rutas(datos, rutas, valores, salida):
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, k, color in zip(ejes, [0, -1], ["#2980b9", "#c0392b"]):
        dibujar_contorno(ax)
        dibujar_ruta(ax, datos, rutas[k], color=color)
        ax.plot(datos.lon, datos.lat, "o", color="#2c3e50", ms=3)
        ax.set_title(f"valor de la hora = {valores[k]:g} €/h")
    fig.tight_layout()
    fig.savefig(salida / "rutas_valor_hora.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
