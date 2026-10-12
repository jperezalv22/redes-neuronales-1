"""
Compara el peaje que calcula el modelo con el precio oficial de viajes completos.

Los pares marcados como "calibración" se usaron para fijar alguna tarifa
(columna eur_km_calibrado de datos/peajes_tarifas.csv), asi que deben dar
casi 0 %. Los de "validación" son una prueba independiente del modelo.

    python validar_peajes.py
"""

import csv

from tsp.datos import CARPETA, cargar_ciudades, cargar_peajes

nombres, _, _ = cargar_ciudades()
peaje, _ = cargar_peajes(len(nombres))

print(f"{'Trayecto':<38}{'Modelo':>9}{'Oficial':>9}{'Error':>9}  Uso")
with open(CARPETA / "peajes_validacion.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        i, j = nombres.index(r["origen"]), nombres.index(r["destino"])
        oficial = float(r["precio_oficial_eur"])
        error = (peaje[i, j] - oficial) / oficial * 100
        print(f"{r['origen'] + ' - ' + r['destino']:<38}{peaje[i, j]:>8.2f}€{oficial:>8.2f}€{error:>+8.1f}%  {r['uso']}")
