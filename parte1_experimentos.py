"""
Parte 1: comparacion de metodos sobre Rosenbrock y Rastrigin en 2D y 3D.

Para cada (funcion, dimension, metodo) se hacen `--corridas` corridas
independientes. La corrida k usa la semilla `--semilla + k`, la misma para
todos los metodos, asi que todo es reproducible.

Todos los metodos reciben el mismo presupuesto, medido en evaluaciones
equivalentes de f:   costo = n_f + 2n * n_grad.

Salidas (en resultados/parte1/):
  - tabla_resultados.csv / .md : media, desviacion, mejor, peor, tasa de exito,
                                 evaluaciones de f y de grad f.
  - convergencia.png           : mediana del mejor valor vs evaluaciones.
  - cajas_valor_final.png      : diagrama de cajas del valor final.

Ejemplos:
    python parte1_experimentos.py                       # experimento completo (~2 min)
    python parte1_experimentos.py --corridas 5          # demo rapida
    python parte1_experimentos.py --eta 5e-4 --presupuesto 20000
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optim import NOMBRES, Problema, correr_metodo


def leer_argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--funciones", nargs="+", default=["rosenbrock", "rastrigin"])
    p.add_argument("--dims", nargs="+", type=int, default=[2, 3])
    p.add_argument("--metodos", nargs="+", default=["gd", "ga", "pso", "de"])
    p.add_argument("--corridas", type=int, default=30)
    p.add_argument("--presupuesto", type=int, default=10000,
                   help="evaluaciones equivalentes de f por corrida")
    p.add_argument("--semilla", type=int, default=2026)
    p.add_argument("--umbral", type=float, default=1e-3,
                   help="una corrida es exitosa si f - f* < umbral")
    # hiperparametros
    p.add_argument("--eta", type=float, default=1e-3, help="tasa de aprendizaje (GD)")
    p.add_argument("--tam-pob", type=int, default=30, help="tamano de poblacion (GA y DE)")
    p.add_argument("--prob-mut", type=float, default=0.1, help="probabilidad de mutacion por gen (GA)")
    p.add_argument("--particulas", type=int, default=30, help="numero de particulas (PSO)")
    p.add_argument("--w", type=float, default=0.7298, help="inercia (PSO)")
    p.add_argument("--c1", type=float, default=1.49618, help="coef. cognitivo (PSO)")
    p.add_argument("--c2", type=float, default=1.49618, help="coef. social (PSO)")
    p.add_argument("--F", type=float, default=0.8, help="factor de mutacion (DE)")
    p.add_argument("--CR", type=float, default=0.9, help="prob. de cruce (DE)")
    p.add_argument("--salida", default="resultados/parte1")
    return p.parse_args()


def parametros_de(metodo, a):
    """Saca de los argumentos los hiperparametros que usa cada metodo."""
    return {
        "gd": {"eta": a.eta},
        "ga": {"tam_pob": a.tam_pob, "prob_mut": a.prob_mut},
        "pso": {"n_particulas": a.particulas, "w": a.w, "c1": a.c1, "c2": a.c2},
        "de": {"tam_pob": a.tam_pob, "F": a.F, "CR": a.CR},
    }[metodo]


def curva_en_rejilla(curva, rejilla):
    """Pasa la curva (costo, mejor_f) a una rejilla fija de costos para poder
    promediar corridas distintas. Es una funcion escalon: en cada punto de la
    rejilla vale el mejor f encontrado hasta ese costo."""
    costos = np.array([c for c, _ in curva])
    valores = np.array([v for _, v in curva])
    idx = np.searchsorted(costos, rejilla, side="right") - 1
    idx = np.clip(idx, 0, len(valores) - 1)
    return valores[idx]


def main():
    a = leer_argumentos()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    rejilla = np.linspace(0, a.presupuesto, 200)

    filas = []
    curvas = {}    # (funcion, dim, metodo) -> matriz corridas x rejilla
    finales = {}   # (funcion, dim, metodo) -> lista de valores finales
    inicio = time.time()

    for fn in a.funciones:
        for dim in a.dims:
            for metodo in a.metodos:
                vals, nfs, ngs, cs = [], [], [], []
                matriz = []
                for k in range(a.corridas):
                    rng = np.random.default_rng(a.semilla + k)
                    prob = Problema(fn, dim)
                    r = correr_metodo(metodo, prob, rng, a.presupuesto, parametros_de(metodo, a))
                    vals.append(r["f"] - prob.f_optimo)
                    nfs.append(prob.n_f)
                    ngs.append(prob.n_grad)
                    cs.append(prob.costo())
                    matriz.append(curva_en_rejilla(r["curva"], rejilla))

                vals = np.array(vals)
                curvas[(fn, dim, metodo)] = np.array(matriz)
                finales[(fn, dim, metodo)] = vals
                fila = {
                    "funcion": fn, "dim": dim, "metodo": metodo,
                    "media": vals.mean(), "desv": vals.std(ddof=1),
                    "mejor": vals.min(), "peor": vals.max(),
                    "exito_%": 100 * np.mean(vals < a.umbral),
                    "evals_f": np.mean(nfs), "evals_grad": np.mean(ngs),
                    "costo_equiv": np.mean(cs),
                }
                filas.append(fila)
                print(f"{fn:10s} {dim}D {metodo:4s} media={fila['media']:.3e} "
                      f"exito={fila['exito_%']:5.1f}%  f={fila['evals_f']:.0f} grad={fila['evals_grad']:.0f}")

    guardar_tablas(filas, salida, a)
    graficar_convergencia(curvas, rejilla, a, salida)
    graficar_cajas(finales, a, salida)
    print(f"\nListo en {time.time() - inicio:.1f} s. Resultados en {salida}/")


def guardar_tablas(filas, salida, a):
    columnas = list(filas[0].keys())
    with open(salida / "tabla_resultados.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columnas, lineterminator="\n")
        w.writeheader()
        w.writerows(filas)

    # version en markdown para pegar en el reporte
    lineas = [
        f"Corridas: {a.corridas}, presupuesto: {a.presupuesto} evaluaciones equivalentes, "
        f"umbral de exito: f - f* < {a.umbral:g}, semilla base: {a.semilla}\n",
        "| Función | Dim | Método | Media | Desv. est. | Mejor | Peor | Éxito (%) | Evals f | Evals ∇f |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in filas:
        lineas.append(
            f"| {r['funcion']} | {r['dim']} | {NOMBRES[r['metodo']]} | {r['media']:.2e} | {r['desv']:.2e} | "
            f"{r['mejor']:.2e} | {r['peor']:.2e} | {r['exito_%']:.0f} | {r['evals_f']:.0f} | {r['evals_grad']:.0f} |")
    (salida / "tabla_resultados.md").write_text("\n".join(lineas) + "\n")


def graficar_convergencia(curvas, rejilla, a, salida):
    fig, ejes = plt.subplots(len(a.funciones), len(a.dims), figsize=(5 * len(a.dims), 4 * len(a.funciones)),
                             squeeze=False)
    for i, fn in enumerate(a.funciones):
        for j, dim in enumerate(a.dims):
            ax = ejes[i, j]
            for metodo in a.metodos:
                m = np.maximum(curvas[(fn, dim, metodo)], 1e-16)  # para poder usar escala log
                med = np.median(m, axis=0)
                q1, q3 = np.percentile(m, [25, 75], axis=0)
                ax.plot(rejilla, med, label=NOMBRES[metodo])
                ax.fill_between(rejilla, q1, q3, alpha=0.2)
            ax.set_yscale("log")
            ax.set_title(f"{fn.capitalize()} {dim}D")
            ax.set_xlabel("evaluaciones equivalentes de f")
            ax.set_ylabel("mejor f - f* (mediana)")
            ax.grid(alpha=0.3)
    ejes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(salida / "convergencia.png", dpi=120)
    plt.close(fig)


def graficar_cajas(finales, a, salida):
    fig, ejes = plt.subplots(len(a.funciones), len(a.dims), figsize=(5 * len(a.dims), 4 * len(a.funciones)),
                             squeeze=False)
    for i, fn in enumerate(a.funciones):
        for j, dim in enumerate(a.dims):
            ax = ejes[i, j]
            datos = [np.maximum(finales[(fn, dim, m)], 1e-16) for m in a.metodos]
            ax.boxplot(datos, tick_labels=[m.upper() for m in a.metodos])
            ax.set_yscale("log")
            ax.set_title(f"{fn.capitalize()} {dim}D")
            ax.set_ylabel("valor final f - f*")
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(salida / "cajas_valor_final.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
