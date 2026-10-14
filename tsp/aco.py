"""
Colonia de hormigas: Ant System (Dorigo, Maniezzo y Colorni, 1996).

Cada hormiga construye un recorrido completo. Estando en la ciudad i, elige
la siguiente ciudad j (no visitada) con probabilidad

    p(i, j) = tau[i,j]^alfa * eta[i,j]^beta / (suma sobre las no visitadas)

donde tau es la feromona y eta = 1 / C[i,j] es la "visibilidad" (preferir
trayectos baratos).

Al final de cada iteracion:
    tau <- (1 - rho) * tau                       (evaporacion)
    tau[i,j] += Q / L_k  para cada arista (i,j) del recorrido de la hormiga k
                         (las hormigas con recorridos baratos dejan mas feromona)

Opcion `dos_opt`: el mejor recorrido de cada iteracion se mejora con 2-opt
antes de compararlo con el mejor global.
"""

import numpy as np

from .utilidades import costo_ruta, dos_opt


def vecino_mas_cercano(C, inicio=0):
    """Recorrido goloso: siempre ir a la ciudad mas barata no visitada."""
    n = len(C)
    ruta = [inicio]
    libres = set(range(n)) - {inicio}
    while libres:
        i = ruta[-1]
        j = min(libres, key=lambda k: C[i, k])
        ruta.append(j)
        libres.remove(j)
    return ruta


def colonia_hormigas(C, rng, n_hormigas=30, iteraciones=200, alfa=1.0, beta=3.0,
                     rho=0.1, Q=1.0, usar_dos_opt=True):
    n = len(C)
    C_sin_cero = C + np.eye(n)  # evitar dividir por cero en la diagonal
    eta = 1.0 / C_sin_cero

    # feromona inicial recomendada: m / L_nn (Dorigo y Stutzle, 2004)
    L_nn = costo_ruta(vecino_mas_cercano(C), C)
    tau = np.full((n, n), n_hormigas / L_nn)

    mejor_ruta, mejor_costo = None, np.inf
    historia = []  # (iteracion, costo, ruta) cada vez que mejora

    for it in range(iteraciones):
        peso = (tau ** alfa) * (eta ** beta)
        rutas, costos = [], []
        for _ in range(n_hormigas):
            ruta = [rng.integers(n)]
            visitada = np.zeros(n, dtype=bool)
            visitada[ruta[0]] = True
            for _ in range(n - 1):
                p = peso[ruta[-1]].copy()
                p[visitada] = 0.0
                # ruleta: lo mismo que rng.choice(n, p=p / p.sum()) (mismo numero
                # aleatorio y mismo resultado), pero sin las validaciones de
                # rng.choice, que eran la mayor parte del tiempo de la colonia
                cdf = (p / p.sum()).cumsum()
                cdf /= cdf[-1]
                j = int(cdf.searchsorted(rng.random(), side="right"))
                ruta.append(j)
                visitada[j] = True
            rutas.append(ruta)
            costos.append(costo_ruta(ruta, C))

        # evaporacion y deposito de feromona
        tau *= (1 - rho)
        for ruta, L in zip(rutas, costos):
            a = np.array(ruta)
            b = np.roll(a, -1)
            tau[a, b] += Q / L
            tau[b, a] += Q / L  # el problema es simetrico

        k = int(np.argmin(costos))
        candidata = rutas[k]
        if usar_dos_opt:
            candidata = dos_opt(candidata, C)
        c = costo_ruta(candidata, C)
        if c < mejor_costo - 1e-9:
            mejor_ruta, mejor_costo = list(candidata), c
            historia.append((it, mejor_costo, mejor_ruta))

    return {"ruta": mejor_ruta, "costo": mejor_costo, "historia": historia}
