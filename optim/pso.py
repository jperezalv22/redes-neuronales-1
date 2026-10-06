"""
Optimizacion por enjambre de particulas (PSO) con peso de inercia.

Cada particula i tiene posicion x_i, velocidad v_i y recuerda su mejor
posicion p_i. El enjambre recuerda la mejor posicion global g.

    v_i <- w v_i + c1 r1 (p_i - x_i) + c2 r2 (g - x_i)
    x_i <- x_i + v_i

r1 y r2 son vectores uniformes en [0, 1]. Los valores por defecto
w = 0.7298, c1 = c2 = 1.49618 son los de Clerc y Kennedy (2002).
La velocidad se limita a `vmax` por componente y la posicion al dominio.
"""

import numpy as np


def pso(problema, rng, presupuesto, n_particulas=30, w=0.7298, c1=1.49618,
        c2=1.49618, vmax_frac=0.2):
    ancho = problema.sup - problema.inf
    vmax = vmax_frac * ancho

    x = problema.punto_aleatorio(rng, n_particulas)
    v = rng.uniform(-vmax, vmax, size=x.shape)
    fx = problema.f(x)

    p, fp = x.copy(), fx.copy()  # mejores personales
    ig = np.argmin(fp)
    g, fg = p[ig].copy(), fp[ig]  # mejor global

    curva = [(problema.costo(), fg)]
    trayectoria = [x.copy()]

    while problema.costo() + n_particulas <= presupuesto:
        r1 = rng.random(x.shape)
        r2 = rng.random(x.shape)
        v = w * v + c1 * r1 * (p - x) + c2 * r2 * (g - x)
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, problema.inf, problema.sup)
        fx = problema.f(x)

        mejora = fx < fp
        p[mejora], fp[mejora] = x[mejora], fx[mejora]
        ig = np.argmin(fp)
        if fp[ig] < fg:
            g, fg = p[ig].copy(), fp[ig]

        curva.append((problema.costo(), fg))
        trayectoria.append(x.copy())

    return {"x": g, "f": fg, "curva": curva, "trayectoria": trayectoria}
