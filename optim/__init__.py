"""Parte 1: funciones de prueba y metodos de optimizacion continua."""

from .funciones import Problema, FUNCIONES
from .gradiente import descenso_gradiente
from .evolutivo import algoritmo_genetico
from .pso import pso
from .evolucion_diferencial import evolucion_diferencial


def correr_metodo(metodo, problema, rng, presupuesto, params):
    """Corre un metodo por nombre. `params` son los hiperparametros extra.

    Todos los metodos arrancan con el mismo generador `rng`, asi que la
    corrida k de cada metodo es reproducible con la semilla.
    """
    if metodo == "gd":
        x0 = problema.punto_aleatorio(rng)
        return descenso_gradiente(problema, x0, presupuesto, **params)
    if metodo == "ga":
        return algoritmo_genetico(problema, rng, presupuesto, **params)
    if metodo == "pso":
        return pso(problema, rng, presupuesto, **params)
    if metodo == "de":
        return evolucion_diferencial(problema, rng, presupuesto, **params)
    raise ValueError(f"metodo desconocido: {metodo}")


NOMBRES = {
    "gd": "Descenso por gradiente",
    "ga": "Algoritmo genético",
    "pso": "PSO",
    "de": "Evolución diferencial",
}
