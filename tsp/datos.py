"""
Carga de los datos del TSP y construccion de la matriz de costos.

El costo de ir de la ciudad i a la j (en euros) es:

    C[i, j] = valor_hora * tiempo[i, j]                    (tiempo del vendedor)
            + peaje[i, j]                                  (autopistas de pago)
            + distancia[i, j] * consumo / 100 * precio     (combustible)

Todos los archivos estan en datos/ y los genera descargar_datos_tsp.py.
"""

import csv
import json
from pathlib import Path

import numpy as np

CARPETA = Path(__file__).resolve().parent.parent / "datos"


def _leer_matriz(nombre):
    with open(CARPETA / nombre) as f:
        return np.array([[float(x) for x in fila] for fila in csv.reader(f)])


def cargar_ciudades():
    """Devuelve (nombres, lat, lon)."""
    with open(CARPETA / "capitales.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    nombres = [r["ciudad"] for r in filas]
    lat = np.array([float(r["lat"]) for r in filas])
    lon = np.array([float(r["lon"]) for r in filas])
    return nombres, lat, lon


def tarifas_por_km():
    """Euros por km de cada via de peaje.

    Para las vias con tarifa verificada se calcula importe / longitud.
    Las demas usan el promedio de las verificadas.
    """
    tarifas = {}
    sin_dato = []
    with open(CARPETA / "peajes_tarifas.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["importe_ligeros_eur"]:
                tarifas[r["via"]] = float(r["importe_ligeros_eur"]) / float(r["longitud_km_aprox"])
            else:
                sin_dato.append(r["via"])
    promedio = float(np.mean(list(tarifas.values())))
    for via in sin_dato:
        tarifas[via] = promedio
    return tarifas


def es_tramo_de_peaje(via, lat):
    """Decide si un tramo de via se paga.

    La AP-7 y la AP-1 son de pago solo en algunos tramos, asi que usamos la
    latitud del tramo para distinguirlos:
      - AP-7: de pago solo al sur de Alicante (lat < 38.25).
      - AP-1: de pago solo entre Vitoria y Eibar (lat > 42.8).
    """
    if via == "AP-7":
        return lat < 38.25
    if via == "AP-1":
        return lat > 42.8
    return True


def via_de_peaje(ref, nombre, tarifas):
    """Devuelve la autopista de pago por la que va un tramo, o None.

    `ref` puede traer varias vias separadas por ";" (por ejemplo "AP-7;E-15").
    Ademas, en OpenStreetMap algunos tramos de autopista de pago estan
    etiquetados sin la P: el tramo Campomanes-Leon de la AP-66 aparece como
    "A-66" con nombre "Autopista Ruta de la Plata". Por eso, si la via es
    "A-xx", existe la "AP-xx" de pago y el nombre dice "Autopista", la
    contamos como AP-xx.
    """
    for via in (v.strip() for v in ref.split(";")):
        if via in tarifas:
            return via
        if via.startswith("A-") and "AP-" + via[2:] in tarifas and "autopista" in nombre.lower():
            return "AP-" + via[2:]
    return None


def cargar_peajes(n):
    """Matriz n x n con el costo de peajes (euros) de cada trayecto."""
    tarifas = tarifas_por_km()
    peaje = np.zeros((n, n))
    km_peaje = np.zeros((n, n))
    with open(CARPETA / "km_por_via.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            via = via_de_peaje(r["ref"], r["nombre"], tarifas)
            if via and es_tramo_de_peaje(via, float(r["lat"])):
                i, j, km = int(r["i"]), int(r["j"]), float(r["km"])
                peaje[i, j] += km * tarifas[via]
                km_peaje[i, j] += km
    # las rutas se descargaron solo para i < j: copiamos al otro lado
    return peaje + peaje.T, km_peaje + km_peaje.T


def cargar_config(ruta="configuracion_tsp.json"):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


class DatosTSP:
    """Agrupa todo lo necesario para el TSP.

    Atributos: nombres, lat, lon, distancia (km), tiempo (h), peaje (EUR),
    combustible (EUR) y el metodo costo(valor_hora) que arma la matriz final.
    """

    def __init__(self, consumo_l_100km, precio_eur_l):
        self.nombres, self.lat, self.lon = cargar_ciudades()
        self.n = len(self.nombres)
        self.distancia = _leer_matriz("distancia_km.csv")
        self.tiempo = _leer_matriz("tiempo_h.csv")
        self.peaje, self.km_peaje = cargar_peajes(self.n)
        self.combustible = self.distancia * consumo_l_100km / 100 * precio_eur_l

    def costo(self, valor_hora):
        return valor_hora * self.tiempo + self.peaje + self.combustible

    def desglose(self, ruta, valor_hora):
        """Totales de un recorrido cerrado: km, horas, peajes, combustible, costo."""
        i = np.array(ruta)
        j = np.roll(i, -1)  # la ciudad siguiente (la ultima vuelve a la primera)
        horas = self.tiempo[i, j].sum()
        return {
            "km": self.distancia[i, j].sum(),
            "horas": horas,
            "peajes_eur": self.peaje[i, j].sum(),
            "combustible_eur": self.combustible[i, j].sum(),
            "tiempo_eur": valor_hora * horas,
            "costo_total_eur": self.costo(valor_hora)[i, j].sum(),
        }
