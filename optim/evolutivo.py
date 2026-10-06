"""
Algoritmo genetico con codificacion real (un individuo = un vector x).

Cada generacion:
  1. Seleccion por torneo: se toman `k` individuos al azar y gana el de menor f.
  2. Cruce BLX-alfa: cada gen del hijo se toma uniforme en el intervalo
     [min - alfa*d, max + alfa*d], con d = |p1 - p2|.  Con probabilidad
     1 - prob_cruce el hijo es copia del primer padre.
  3. Mutacion gaussiana: a cada gen, con probabilidad prob_mut, se le suma
     N(0, sigma^2), con sigma = escala_mut * (ancho del dominio).
  4. Elitismo: el mejor individuo pasa sin cambios a la siguiente generacion.
"""

import numpy as np


def torneo(pob, fit, rng, k=3):
    idx = rng.integers(0, len(pob), size=k)
    return pob[idx[np.argmin(fit[idx])]]


def algoritmo_genetico(problema, rng, presupuesto, tam_pob=40, prob_cruce=0.9,
                       alfa=0.5, prob_mut=0.1, escala_mut=0.1):
    ancho = problema.sup - problema.inf
    pob = problema.punto_aleatorio(rng, tam_pob)
    fit = problema.f(pob)

    i_mejor = np.argmin(fit)
    curva = [(problema.costo(), fit[i_mejor])]
    trayectoria = [pob.copy()]

    while problema.costo() + (tam_pob - 1) <= presupuesto:
        hijos = [pob[i_mejor].copy()]  # elitismo
        while len(hijos) < tam_pob:
            p1, p2 = torneo(pob, fit, rng), torneo(pob, fit, rng)
            if rng.random() < prob_cruce:
                bajo, alto = np.minimum(p1, p2), np.maximum(p1, p2)
                d = alto - bajo
                hijo = rng.uniform(bajo - alfa * d, alto + alfa * d)
            else:
                hijo = p1.copy()
            muta = rng.random(problema.dim) < prob_mut
            hijo[muta] += rng.normal(0, escala_mut * ancho[muta])
            hijos.append(np.clip(hijo, problema.inf, problema.sup))

        pob = np.array(hijos)
        # el elite ya tiene su f calculada: solo evaluamos a los nuevos
        f_elite = fit[i_mejor]
        fit = np.concatenate([[f_elite], problema.f(pob[1:])])
        i_mejor = np.argmin(fit)

        curva.append((problema.costo(), fit[i_mejor]))
        trayectoria.append(pob.copy())

    return {"x": pob[i_mejor], "f": fit[i_mejor], "curva": curva, "trayectoria": trayectoria}
