"""
Descenso por gradiente "clasico" con tasa de aprendizaje fija.

Regla de actualizacion:
    x_{k+1} = x_k - eta * grad f(x_k)

Para y cuando:
  - se acaba el presupuesto de evaluaciones equivalentes, o
  - la norma del gradiente es menor que `tol` (llegamos a un punto critico), o
  - el valor se vuelve infinito/NaN (la tasa de aprendizaje era muy grande).

El descenso por gradiente en si no necesita evaluar f, solo el gradiente.
Aun asi evaluamos f en cada iteracion para poder dibujar la curva de
convergencia; esas evaluaciones tambien se cuentan (n_f), no se esconden.
"""

import numpy as np


def descenso_gradiente(problema, x0, presupuesto, eta=1e-3, tol=1e-8):
    x = np.array(x0, dtype=float)
    fx = problema.f(x)
    mejor_x, mejor_f = x.copy(), fx

    curva = [(problema.costo(), mejor_f)]  # (evaluaciones equivalentes, mejor f)
    trayectoria = [x.copy()]

    # costo de una iteracion = 1 gradiente (2n) + 1 evaluacion de f
    costo_iter = 2 * problema.dim + 1
    # si diverge, numpy avisa "overflow"; lo silenciamos porque la
    # divergencia la detectamos nosotros con np.isfinite
    with np.errstate(over="ignore", invalid="ignore"):
        while problema.costo() + costo_iter <= presupuesto:
            g = problema.grad(x)
            if np.linalg.norm(g) < tol:
                break
            x = x - eta * g
            fx = problema.f(x)
            if not np.isfinite(fx):
                break  # divergio
            if fx < mejor_f:
                mejor_x, mejor_f = x.copy(), fx
            curva.append((problema.costo(), mejor_f))
            trayectoria.append(x.copy())

    return {"x": mejor_x, "f": mejor_f, "curva": curva, "trayectoria": trayectoria}
