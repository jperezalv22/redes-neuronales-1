"""
Funciones de prueba de la Parte 1: Rosenbrock y Rastrigin.

Escogimos estas dos porque son "opuestas":
  - Rosenbrock tiene un solo minimo, pero esta en un valle largo, curvo y
    estrecho (mal condicionado). El gradiente apunta casi siempre hacia las
    paredes del valle y no hacia el minimo.
  - Rastrigin tiene muchisimos minimos locales (una rejilla de "huecos")
    encima de una parabola. El gradiente se queda en el hueco mas cercano.

Las dos estan definidas para cualquier dimension n, asi que sirven para 2D y 3D.

La clase Problema envuelve la funcion y CUENTA cuantas veces se evalua f y
cuantas veces se evalua el gradiente. Esos contadores son la base de la
comparacion justa entre metodos.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Rosenbrock
#   f(x) = sum_{i=1}^{n-1} [ 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
#   dominio usado: [-2.048, 2.048]^n
#   minimo global: f(1, ..., 1) = 0
# ---------------------------------------------------------------------------
def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    a = x[..., :-1]
    b = x[..., 1:]
    return np.sum(100.0 * (b - a**2) ** 2 + (1.0 - a) ** 2, axis=-1)


def grad_rosenbrock(x):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    a = x[:-1]
    b = x[1:]
    # derivada respecto a x_i cuando x_i aparece como "a" (primer termino)
    g[:-1] += -400.0 * a * (b - a**2) - 2.0 * (1.0 - a)
    # derivada respecto a x_{i+1} cuando aparece como "b"
    g[1:] += 200.0 * (b - a**2)
    return g


# ---------------------------------------------------------------------------
# Rastrigin
#   f(x) = 10 n + sum_i [ x_i^2 - 10 cos(2 pi x_i) ]
#   dominio usado: [-5.12, 5.12]^n
#   minimo global: f(0, ..., 0) = 0
# ---------------------------------------------------------------------------
def rastrigin(x):
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    return 10.0 * n + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x), axis=-1)


def grad_rastrigin(x):
    x = np.asarray(x, dtype=float)
    return 2.0 * x + 20.0 * np.pi * np.sin(2 * np.pi * x)


# Informacion de cada funcion: (f, gradiente, limite del dominio, punto optimo)
FUNCIONES = {
    "rosenbrock": (rosenbrock, grad_rosenbrock, 2.048, 1.0),
    "rastrigin": (rastrigin, grad_rastrigin, 5.12, 0.0),
}


def gradiente_numerico(f, x, h=1e-6):
    """Gradiente por diferencias centrales. Cuesta 2n evaluaciones de f.

    Se usa para verificar los gradientes analiticos (ver pruebas.py) y es
    la base de la equivalencia 1 gradiente = 2n evaluaciones de f.
    """
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


class Problema:
    """Una funcion de prueba en una dimension dada, con contadores.

    Atributos importantes:
        n_f     -> cuantas veces se ha evaluado f (un punto = 1 evaluacion)
        n_grad  -> cuantas veces se ha evaluado el gradiente
    """

    def __init__(self, nombre, dim):
        self.nombre = nombre
        self.dim = dim
        self._f, self._grad, self.limite, optimo = FUNCIONES[nombre]
        self.x_optimo = np.full(dim, optimo)
        self.f_optimo = 0.0
        self.inf = np.full(dim, -self.limite)
        self.sup = np.full(dim, self.limite)
        self.n_f = 0
        self.n_grad = 0

    def f(self, x):
        """Evalua f en un punto (vector) o en varios (matriz, un punto por fila)."""
        x = np.asarray(x, dtype=float)
        self.n_f += 1 if x.ndim == 1 else x.shape[0]
        return self._f(x)

    def grad(self, x):
        self.n_grad += 1
        return self._grad(x)

    def costo(self):
        """Costo total en 'evaluaciones equivalentes de f'.

        Equivalencia: 1 gradiente = 2n evaluaciones de f, que es lo que
        costaria aproximarlo con diferencias centrales.
        """
        return self.n_f + 2 * self.dim * self.n_grad

    def punto_aleatorio(self, rng, cantidad=None):
        """Punto(s) uniformes dentro del dominio."""
        forma = self.dim if cantidad is None else (cantidad, self.dim)
        return rng.uniform(self.inf, self.sup, size=forma)
