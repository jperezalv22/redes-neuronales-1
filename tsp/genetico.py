"""
Algoritmo genetico para el TSP con representacion de permutacion.

Un individuo es una lista con el orden en que se visitan las ciudades.

  - Seleccion: torneo de tamano k.
  - Cruce OX (order crossover, Davis 1985): se copia un pedazo del padre 1
    y el resto de ciudades se completa en el orden en que aparecen en el
    padre 2. Asi el hijo siempre es una permutacion valida.
  - Mutacion por inversion: se invierte un pedazo al azar de la ruta.
  - Elitismo: los `elite` mejores pasan directo a la siguiente generacion.

Opcion `dos_opt`: cada hijo nuevo se mejora con 2-opt con probabilidad
`prob_2opt`. Esto lo convierte en un algoritmo "memetico" (genetico + busqueda
local). Sin esa ayuda el genetico necesita muchisimas mas generaciones.
"""

import numpy as np

from .utilidades import costo_ruta, dos_opt


def cruce_ox(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    hijo = [None] * n
    hijo[a:b + 1] = p1[a:b + 1]
    usadas = set(hijo[a:b + 1])
    resto = [c for c in p2 if c not in usadas]
    k = 0
    for i in list(range(b + 1, n)) + list(range(0, a)):
        hijo[i] = resto[k]
        k += 1
    return hijo


def mutacion_inversion(ruta, rng):
    a, b = sorted(rng.choice(len(ruta), size=2, replace=False))
    ruta[a:b + 1] = ruta[a:b + 1][::-1]
    return ruta


def algoritmo_genetico_tsp(C, rng, tam_pob=100, generaciones=300, prob_cruce=0.9,
                           prob_mut=0.3, torneo=5, elite=2, usar_dos_opt=True, prob_2opt=0.05):
    n = len(C)
    pob = [list(rng.permutation(n)) for _ in range(tam_pob)]
    costos = np.array([costo_ruta(r, C) for r in pob])

    mejor_ruta, mejor_costo = None, np.inf
    historia = []

    for gen in range(generaciones):
        orden = np.argsort(costos)
        if costos[orden[0]] < mejor_costo - 1e-9:
            mejor_ruta, mejor_costo = list(pob[orden[0]]), costos[orden[0]]
            historia.append((gen, mejor_costo, mejor_ruta))

        def elegir():
            idx = rng.choice(tam_pob, size=torneo, replace=False)
            return pob[idx[np.argmin(costos[idx])]]

        nueva = [list(pob[i]) for i in orden[:elite]]
        while len(nueva) < tam_pob:
            p1, p2 = elegir(), elegir()
            hijo = cruce_ox(p1, p2, rng) if rng.random() < prob_cruce else list(p1)
            if rng.random() < prob_mut:
                hijo = mutacion_inversion(hijo, rng)
            if usar_dos_opt and rng.random() < prob_2opt:
                hijo = dos_opt(hijo, C)
            nueva.append(hijo)

        pob = nueva
        costos = np.array([costo_ruta(r, C) for r in pob])

    i = int(np.argmin(costos))
    if costos[i] < mejor_costo - 1e-9:
        mejor_ruta, mejor_costo = list(pob[i]), costos[i]
        historia.append((generaciones, mejor_costo, mejor_ruta))
    return {"ruta": mejor_ruta, "costo": mejor_costo, "historia": historia}
