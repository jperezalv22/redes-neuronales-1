"""Funciones comunes para el TSP: costo de un recorrido y busqueda local 2-opt."""

import numpy as np


def costo_ruta(ruta, C):
    """Costo de un recorrido cerrado (se vuelve a la ciudad de salida)."""
    ruta = np.asarray(ruta)
    return C[ruta, np.roll(ruta, -1)].sum()


def dos_opt(ruta, C):
    """Mejora un recorrido con 2-opt hasta que no haya mejora posible.

    2-opt quita dos aristas (a-b) y (c-d) y reconecta como (a-c) y (b-d),
    lo que equivale a invertir el pedazo de ruta entre b y c.
    Se acepta el cambio si baja el costo.

    Ojo: esta formula supone costos simetricos (C[i,j] == C[j,i]), que es
    nuestro caso porque cada ruta se descargo una sola vez para i < j.
    """
    ruta = np.array(ruta)
    n = len(ruta)
    mejoro = True
    while mejoro:
        mejoro = False
        for i in range(n - 2):
            a, b = ruta[i], ruta[i + 1]
            # todos los j posibles a la vez (con numpy) en lugar de un for
            j = np.arange(i + 2, n if i > 0 else n - 1)
            c, d = ruta[j], ruta[(j + 1) % n]
            delta = C[a, c] + C[b, d] - C[a, b] - C[c, d]
            k = np.argmin(delta)
            if delta[k] < -1e-9:
                jk = j[k]
                ruta[i + 1 : jk + 1] = ruta[i + 1 : jk + 1][::-1]
                mejoro = True
    return list(ruta)
