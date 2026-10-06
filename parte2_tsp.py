"""
Parte 2: recorrido por las 47 capitales de la Espana peninsular.

Resuelve el TSP con colonia de hormigas (aco) y/o algoritmo genetico (ga),
y guarda en resultados/parte2/:
  - ruta_<metodo>.png        : mejor recorrido sobre el mapa
  - evolucion_<metodo>.gif   : como mejora la solucion (con --gif)
  - resumen_<metodo>.md      : desglose de costos y orden de las ciudades
  - convergencia_tsp.png     : mejor costo por iteracion

Los parametros del vehiculo y el valor de la hora se leen de
configuracion_tsp.json y se pueden cambiar con las opciones de abajo.

Ejemplos:
    python parte2_tsp.py --gif                       # demo (aco y ga, < 1 min)
    python parte2_tsp.py --metodo aco --valor-hora 50 --rho 0.3
    python parte2_tsp.py --metodo ga --poblacion 200 --sin-2opt
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsp import DatosTSP, algoritmo_genetico_tsp, cargar_config, colonia_hormigas
from tsp.mapa import figura_ruta, gif_evolucion

NOMBRES = {"aco": "Colonia de hormigas", "ga": "Algoritmo genético"}


def leer_argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configuracion_tsp.json")
    p.add_argument("--metodo", choices=["aco", "ga", "ambos"], default="ambos")
    p.add_argument("--semilla", type=int, default=2026)
    p.add_argument("--valor-hora", type=float, help="EUR por hora del vendedor (sobrescribe el config)")
    p.add_argument("--consumo", type=float, help="litros / 100 km (sobrescribe el config)")
    p.add_argument("--precio", type=float, help="EUR / litro (sobrescribe el config)")
    p.add_argument("--sin-2opt", action="store_true", help="no usar busqueda local 2-opt")
    p.add_argument("--gif", action="store_true", help="generar el GIF de la evolucion")
    # colonia de hormigas
    p.add_argument("--hormigas", type=int, default=30)
    p.add_argument("--iteraciones", type=int, default=150)
    p.add_argument("--alfa", type=float, default=1.0, help="peso de la feromona")
    p.add_argument("--beta", type=float, default=3.0, help="peso de la visibilidad 1/C")
    p.add_argument("--rho", type=float, default=0.1, help="tasa de evaporacion")
    # genetico
    p.add_argument("--poblacion", type=int, default=100)
    p.add_argument("--generaciones", type=int, default=300)
    p.add_argument("--prob-cruce", type=float, default=0.9)
    p.add_argument("--prob-mut", type=float, default=0.3)
    p.add_argument("--prob-2opt", type=float, default=0.05, help="prob. de aplicar 2-opt a cada hijo (GA)")
    p.add_argument("--salida", default="resultados/parte2")
    return p.parse_args()


def resolver(metodo, C, a):
    rng = np.random.default_rng(a.semilla)
    if metodo == "aco":
        return colonia_hormigas(C, rng, n_hormigas=a.hormigas, iteraciones=a.iteraciones,
                                alfa=a.alfa, beta=a.beta, rho=a.rho,
                                usar_dos_opt=not a.sin_2opt)
    return algoritmo_genetico_tsp(C, rng, tam_pob=a.poblacion, generaciones=a.generaciones,
                                  prob_cruce=a.prob_cruce, prob_mut=a.prob_mut,
                                  usar_dos_opt=not a.sin_2opt, prob_2opt=a.prob_2opt)


def escribir_resumen(archivo, metodo, datos, res, valor_hora, cfg, segundos):
    d = datos.desglose(res["ruta"], valor_hora)
    orden = " → ".join(datos.nombres[k] for k in res["ruta"] + [res["ruta"][0]])
    texto = f"""# {NOMBRES[metodo]}

- Valor de la hora: {valor_hora:.2f} €/h
- Vehículo: {cfg['vehiculo']} ({cfg['consumo_l_100km']} l/100 km, {cfg['precio_combustible_eur_l']} €/l)
- Tiempo de cómputo: {segundos:.1f} s

| Concepto | Valor |
|---|---|
| Distancia total | {d['km']:,.1f} km |
| Tiempo de conducción | {d['horas']:,.2f} h |
| Costo del tiempo | {d['tiempo_eur']:,.2f} € |
| Peajes | {d['peajes_eur']:,.2f} € |
| Combustible | {d['combustible_eur']:,.2f} € |
| **Costo total** | **{d['costo_total_eur']:,.2f} €** |

Orden del recorrido:

{orden}
"""
    Path(archivo).write_text(texto, encoding="utf-8")
    return d


def main():
    a = leer_argumentos()
    cfg = cargar_config(a.config)
    if a.valor_hora is not None:
        cfg["valor_hora_eur"] = a.valor_hora
    if a.consumo is not None:
        cfg["consumo_l_100km"] = a.consumo
    if a.precio is not None:
        cfg["precio_combustible_eur_l"] = a.precio

    datos = DatosTSP(cfg["consumo_l_100km"], cfg["precio_combustible_eur_l"])
    C = datos.costo(cfg["valor_hora_eur"])
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    metodos = ["aco", "ga"] if a.metodo == "ambos" else [a.metodo]
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in metodos:
        t0 = time.time()
        res = resolver(m, C, a)
        seg = time.time() - t0
        d = escribir_resumen(salida / f"resumen_{m}.md", m, datos, res, cfg["valor_hora_eur"], cfg, seg)
        print(f"{NOMBRES[m]}: costo {res['costo']:,.2f} EUR | {d['km']:,.0f} km | "
              f"{d['horas']:.1f} h | peajes {d['peajes_eur']:.2f} EUR | {seg:.1f} s")

        figura_ruta(datos, res["ruta"], f"{NOMBRES[m]}: {res['costo']:,.2f} €",
                    salida / f"ruta_{m}.png")
        if a.gif:
            gif_evolucion(datos, res["historia"], NOMBRES[m], salida / f"evolucion_{m}.gif")

        its = [h[0] for h in res["historia"]]
        costos = [h[1] for h in res["historia"]]
        ax.step(its, costos, where="post", label=NOMBRES[m])

    ax.set_xlabel("iteración / generación")
    ax.set_ylabel("mejor costo (€)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(salida / "convergencia_tsp.png", dpi=120)
    plt.close(fig)
    print(f"Resultados en {salida}/")


if __name__ == "__main__":
    main()
