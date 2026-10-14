"""Funciones comunes para el TSP: costo de un recorrido y busqueda local 2-opt."""

import numpy as np


def costo_ruta(ruta, C):
    """Costo de un recorrido cerrado (se vuelve a la ciudad de salida)."""
    ruta = np.asarray(ruta)
    siguiente = np.concatenate((ruta[1:], ruta[:1]))  # = np.roll(ruta, -1), mas barato
    return C[ruta, siguiente].sum()


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
    # pares (i, j) validos: j >= i + 2, y para i = 0 no se toma j = n - 1
    # (esas dos aristas son vecinas). Los demas quedan en +inf.
    invalido = ~np.triu(np.ones((n, n), dtype=bool), 2)
    invalido[0, n - 1] = True

    mejoro = True
    while mejoro:
        mejoro = False
        i = 0
        while i < n - 2:
            # delta[i, j] de las filas i..n-3 a la vez. Mientras no haya mejora
            # la ruta no cambia, asi que estas filas sirven para todos los i que
            # siguen: solo se recalculan despues de invertir un pedazo.
            sig = np.concatenate((ruta[1:], ruta[:1]))  # ciudad siguiente
            ab = C[ruta, sig]  # costo de la arista que sale de cada posicion
            f = slice(i, n - 2)
            delta = C[ruta[f, None], ruta] + C[sig[f, None], sig] - ab[f, None] - ab[None, :]
            delta[invalido[f]] = np.inf
            filas = np.flatnonzero(delta.min(axis=1) < -1e-9)
            if len(filas) == 0:
                break
            k = filas[0]
            i += k
            jk = np.argmin(delta[k])
            ruta[i + 1 : jk + 1] = ruta[i + 1 : jk + 1][::-1]
            mejoro = True
            i += 1
    return list(ruta)
