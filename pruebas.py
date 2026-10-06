"""
Verificaciones rapidas de que las funciones y gradientes estan bien.

Las hicimos porque es facil equivocarse derivando (o que una IA se equivoque):
  1. f en el optimo debe dar 0.
  2. El gradiente analitico debe coincidir con diferencias centrales
     en puntos aleatorios.
  3. El gradiente en el optimo debe ser 0.
  4. Los contadores de evaluaciones deben sumar lo que esperamos.

Uso:
    python pruebas.py
"""

import numpy as np

from optim.funciones import FUNCIONES, Problema, gradiente_numerico


def main():
    rng = np.random.default_rng(0)
    for nombre, (f, grad, limite, optimo) in FUNCIONES.items():
        for dim in (2, 3, 5):
            x_opt = np.full(dim, optimo)
            assert abs(f(x_opt)) < 1e-12, f"{nombre}: f(optimo) != 0"
            assert np.allclose(grad(x_opt), 0, atol=1e-9), f"{nombre}: grad(optimo) != 0"

            peor = 0.0
            for _ in range(100):
                x = rng.uniform(-limite, limite, dim)
                ga, gn = grad(x), gradiente_numerico(f, x)
                err = np.max(np.abs(ga - gn)) / (1 + np.max(np.abs(ga)))
                peor = max(peor, err)
            assert peor < 1e-6, f"{nombre} dim {dim}: gradiente no coincide ({peor:.2e})"
            print(f"ok  {nombre:10s} dim={dim}  error relativo max del gradiente = {peor:.1e}")

    # contadores
    p = Problema("rastrigin", 3)
    p.f(np.zeros(3))
    p.f(np.zeros((10, 3)))
    p.grad(np.zeros(3))
    assert (p.n_f, p.n_grad, p.costo()) == (11, 1, 11 + 6)
    print("ok  contadores: 11 evaluaciones de f + 1 gradiente = 17 equivalentes en 3D")


if __name__ == "__main__":
    main()
