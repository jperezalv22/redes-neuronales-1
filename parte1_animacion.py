"""
GIF del proceso de optimizacion en 2D sobre las curvas de nivel de la funcion.

  - Descenso por gradiente: se dibuja el camino que sigue el punto.
  - Metodos de poblacion (pso, de, ga): se dibuja la poblacion en cada iteracion.

Ejemplos:
    python parte1_animacion.py                                 # los 4 GIF del reporte
    python parte1_animacion.py --funcion rastrigin --metodo pso
    python parte1_animacion.py --funcion rosenbrock --metodo gd --eta 5e-4
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from optim import NOMBRES, Problema, correr_metodo


def leer_argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--funcion", choices=["rosenbrock", "rastrigin"], default=None,
                   help="si no se da, se hacen las dos")
    p.add_argument("--metodo", choices=["gd", "ga", "pso", "de"], default=None,
                   help="si no se da, se hacen gd y pso")
    p.add_argument("--presupuesto", type=int, default=3000)
    p.add_argument("--semilla", type=int, default=2)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--cuadros", type=int, default=60, help="numero de cuadros del GIF")
    p.add_argument("--salida", default="resultados/parte1")
    return p.parse_args()


def animar(funcion, metodo, a):
    prob = Problema(funcion, 2)
    rng = np.random.default_rng(a.semilla)
    params = {"eta": a.eta} if metodo == "gd" else {}
    res = correr_metodo(metodo, prob, rng, a.presupuesto, params)
    tray = res["trayectoria"]

    # escogemos `cuadros` iteraciones espaciadas (mas densas al inicio,
    # que es donde pasan mas cosas)
    t = np.linspace(0, 1, a.cuadros) ** 2
    indices = np.unique((t * (len(tray) - 1)).astype(int))

    # curvas de nivel (en log para que se vea el valle de Rosenbrock)
    L = prob.limite
    xs = np.linspace(-L, L, 300)
    X, Y = np.meshgrid(xs, xs)
    Z = prob._f(np.stack([X, Y], axis=-1))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.contourf(X, Y, np.log1p(Z), levels=40, cmap="viridis")
    ax.plot(*prob.x_optimo, "r*", ms=15, label="mínimo global")
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    camino, = ax.plot([], [], "w.-", lw=1, ms=3)
    puntos, = ax.plot([], [], "o", color="orange", mec="k", ms=6)
    ax.legend(loc="upper right")

    def dibujar(c):
        k = indices[c]
        if metodo == "gd":
            pts = np.array(tray[: k + 1])
            camino.set_data(pts[:, 0], pts[:, 1])
            puntos.set_data([pts[-1, 0]], [pts[-1, 1]])
            fk = prob._f(pts[-1])
        else:
            pob = tray[k]
            puntos.set_data(pob[:, 0], pob[:, 1])
            fk = prob._f(pob).min()
        ax.set_title(f"{NOMBRES[metodo]} - {funcion.capitalize()} 2D\n"
                     f"iteración {k}, mejor f = {fk:.3e}")
        return camino, puntos

    anim = FuncAnimation(fig, dibujar, frames=len(indices), blit=False)
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    archivo = Path(a.salida) / f"anim_{funcion}_{metodo}.gif"
    anim.save(archivo, writer=PillowWriter(fps=8))
    plt.close(fig)
    print("guardado", archivo)


def main():
    a = leer_argumentos()
    funciones = [a.funcion] if a.funcion else ["rosenbrock", "rastrigin"]
    metodos = [a.metodo] if a.metodo else ["gd", "pso"]
    for fn in funciones:
        for m in metodos:
            animar(fn, m, a)


if __name__ == "__main__":
    main()
