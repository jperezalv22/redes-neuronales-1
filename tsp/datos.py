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
    """Tramos de pago de cada via: {via: [(lat_min, lat_max, euros_por_km), ...]}.

    La tarifa de cada tramo es importe / longitud oficial. Si la longitud
    oficial no cuadra con la que da OSRM, la tarifa esta calibrada con el
    precio de un viaje completo (columna eur_km_calibrado; ver fuente_longitud).
    Las vias sin tarifa verificada usan el promedio de las demas.

    Algunas vias son de pago solo en parte (la AP-7 tiene tres concesiones al
    sur de Alicante, la AP-1 solo entre Eibar y Vitoria): esos tramos se
    separan por latitud. Sin lat_min/lat_max, se paga toda la via.
    """
    tramos = {}
    sin_dato = []
    with open(CARPETA / "peajes_tarifas.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["eur_km_calibrado"]:
                eur_km = float(r["eur_km_calibrado"])
            elif r["importe_ligeros_eur"]:
                eur_km = float(r["importe_ligeros_eur"]) / float(r["longitud_km"])
            else:
                sin_dato.append(r["via"])
                continue
            lat_min = float(r["lat_min"]) if r["lat_min"] else -90.0
            lat_max = float(r["lat_max"]) if r["lat_max"] else 90.0
            tramos.setdefault(r["via"], []).append((lat_min, lat_max, eur_km))
    promedio = float(np.mean([t[2] for lista in tramos.values() for t in lista]))
    for via in sin_dato:
        tramos[via] = [(-90.0, 90.0, promedio)]
    return tramos


def tarifa_del_tramo(tramos, via, lat):
    """Euros por km de un tramo de `via` a latitud `lat`, o 0 si ahi es libre."""
    for lat_min, lat_max, eur_km in tramos[via]:
        if lat_min <= lat < lat_max:
            return eur_km
    return 0.0


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
            eur_km = tarifa_del_tramo(tarifas, via, float(r["lat"])) if via else 0.0
            if eur_km > 0:
                i, j, km = int(r["i"]), int(r["j"]), float(r["km"])
                peaje[i, j] += km * eur_km
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
