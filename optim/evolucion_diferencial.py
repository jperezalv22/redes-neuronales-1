"""
Evolucion diferencial, variante DE/rand/1/bin (Storn y Price, 1997).

Para cada individuo x_i de la poblacion:
  1. Mutacion: se escogen tres individuos distintos r1, r2, r3 (y distintos de i)
         v = x_r1 + F (x_r2 - x_r3)
  2. Cruce binomial: cada componente del vector de prueba u se toma de v con
     probabilidad CR, si no de x_i. Una componente al azar (j_rand) siempre
     viene de v, para que u no sea igual a x_i.
  3. Seleccion: si f(u) <= f(x_i), u reemplaza a x_i.
"""

import numpy as np


def evolucion_diferencial(problema, rng, presupuesto, tam_pob=30, F=0.8, CR=0.9):
    n, d = tam_pob, problema.dim
    pob = problema.punto_aleatorio(rng, n)
    fit = problema.f(pob)

    curva = [(problema.costo(), fit.min())]
    trayectoria = [pob.copy()]

    while problema.costo() + n <= presupuesto:
        prueba = np.empty_like(pob)
        for i in range(n):
            otros = [k for k in range(n) if k != i]
            r1, r2, r3 = rng.choice(otros, size=3, replace=False)
            v = pob[r1] + F * (pob[r2] - pob[r3])

            cruza = rng.random(d) < CR
            cruza[rng.integers(d)] = True  # j_rand
            u = np.where(cruza, v, pob[i])
            prueba[i] = np.clip(u, problema.inf, problema.sup)

        f_prueba = problema.f(prueba)  # evaluamos toda la generacion junta
        gana = f_prueba <= fit
        pob[gana], fit[gana] = prueba[gana], f_prueba[gana]

        curva.append((problema.costo(), fit.min()))
        trayectoria.append(pob.copy())

    i = np.argmin(fit)
    return {"x": pob[i], "f": fit[i], "curva": curva, "trayectoria": trayectoria}
