"""
Descarga los datos del TSP y los guarda en la carpeta datos/.

Este script es el UNICO que necesita internet. Se corre una sola vez y los
archivos que genera quedan en el repositorio, asi el resto del proyecto
funciona sin conexion.

Fuentes:
  - Coordenadas: Nominatim (OpenStreetMap).
  - Distancia y tiempo por carretera: servidor publico de OSRM
    (router.project-osrm.org), perfil "driving".
  - Kilometros recorridos en cada autopista: se sacan de los pasos (steps)
    de la ruta de OSRM, que traen el codigo de la via (AP-6, AP-68, ...).
  - Contorno de Espana para el mapa: johan/world.geo.json (Natural Earth).

Uso:
    python descargar_datos_tsp.py
"""

import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

CARPETA = Path(__file__).parent / "datos"
AGENTE = "taller-redes-unal/1.0 (curso Redes neuronales, UNAL)"

# 47 capitales de provincia de la Espana peninsular: (ciudad, provincia)
CAPITALES = [
    ("A Coruña", "A Coruña"), ("Lugo", "Lugo"), ("Ourense", "Ourense"),
    ("Pontevedra", "Pontevedra"), ("Oviedo", "Asturias"), ("Santander", "Cantabria"),
    ("Bilbao", "Bizkaia"), ("Donostia-San Sebastián", "Gipuzkoa"),
    ("Vitoria-Gasteiz", "Araba/Álava"), ("Pamplona", "Navarra"),
    ("Logroño", "La Rioja"), ("Zaragoza", "Zaragoza"), ("Huesca", "Huesca"),
    ("Teruel", "Teruel"), ("Barcelona", "Barcelona"), ("Girona", "Girona"),
    ("Lleida", "Lleida"), ("Tarragona", "Tarragona"),
    ("Castelló de la Plana", "Castellón"), ("València", "Valencia"),
    ("Alacant", "Alicante"), ("Murcia", "Murcia"), ("Albacete", "Albacete"),
    ("Cuenca", "Cuenca"), ("Guadalajara", "Guadalajara"), ("Toledo", "Toledo"),
    ("Ciudad Real", "Ciudad Real"), ("Madrid", "Madrid"), ("Segovia", "Segovia"),
    ("Ávila", "Ávila"), ("Salamanca", "Salamanca"), ("Zamora", "Zamora"),
    ("Valladolid", "Valladolid"), ("Palencia", "Palencia"), ("León", "León"),
    ("Burgos", "Burgos"), ("Soria", "Soria"), ("Cáceres", "Cáceres"),
    ("Badajoz", "Badajoz"), ("Huelva", "Huelva"), ("Sevilla", "Sevilla"),
    ("Cádiz", "Cádiz"), ("Córdoba", "Córdoba"), ("Jaén", "Jaén"),
    ("Granada", "Granada"), ("Málaga", "Málaga"), ("Almería", "Almería"),
]


def pedir_json(url):
    """Hace una peticion GET y devuelve la respuesta como diccionario."""
    req = urllib.request.Request(url, headers={"User-Agent": AGENTE})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def geocodificar():
    """Busca latitud y longitud de cada capital en Nominatim."""
    filas = []
    for ciudad, provincia in CAPITALES:
        q = urllib.parse.urlencode({
            "city": ciudad, "country": "Spain", "format": "json", "limit": 1,
        })
        res = pedir_json("https://nominatim.openstreetmap.org/search?" + q)
        lat, lon = float(res[0]["lat"]), float(res[0]["lon"])
        print(f"  {ciudad:25s} {lat:9.4f} {lon:9.4f}")
        filas.append({"ciudad": ciudad, "provincia": provincia, "lat": lat, "lon": lon})
        time.sleep(1.1)  # Nominatim pide maximo una peticion por segundo
    return filas


def ruta(a, b):
    """Pide a OSRM la ruta en carro entre dos ciudades.

    Devuelve (km, horas, tramos) donde cada tramo es [ref, nombre, lat, km]:
    cuantos km de la ruta van por cada via con codigo (AP-6, A-1, ...).
    """
    coords = f"{a['lon']},{a['lat']};{b['lon']},{b['lat']}"
    url = (f"https://router.project-osrm.org/route/v1/driving/{coords}"
           "?overview=false&steps=true")
    for intento in range(5):
        try:
            res = pedir_json(url)
            break
        except Exception as e:  # el servidor publico a veces falla, reintentamos
            print("    error, reintentando:", e)
            time.sleep(5 * (intento + 1))
    r = res["routes"][0]
    tramos = []
    for paso in r["legs"][0]["steps"]:
        ref = paso.get("ref")
        if not ref:
            continue
        # guardamos la referencia completa (puede ser "AP-7;E-15"), el nombre
        # de la via y la latitud (para separar tramos de pago de la AP-7 y AP-1)
        lat = paso["maneuver"]["location"][1]
        tramos.append([ref, paso.get("name", ""), round(lat, 3), round(paso["distance"] / 1000, 3)])
    return r["distance"] / 1000, r["duration"] / 3600, tramos


def main():
    CARPETA.mkdir(exist_ok=True)
    hoy = date.today().isoformat()

    print("1) Coordenadas (Nominatim)")
    ciudades = geocodificar()
    with open(CARPETA / "capitales.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ciudad", "provincia", "lat", "lon"], lineterminator="\n")
        w.writeheader()
        w.writerows(ciudades)

    print("2) Rutas entre cada par de ciudades (OSRM)")
    n = len(ciudades)
    dist = [[0.0] * n for _ in range(n)]
    tiempo = [[0.0] * n for _ in range(n)]
    tramos = []  # filas (i, j, ref, nombre, lat, km)
    for i in range(n):
        for j in range(i + 1, n):
            km, h, pasos = ruta(ciudades[i], ciudades[j])
            dist[i][j] = dist[j][i] = km
            tiempo[i][j] = tiempo[j][i] = h
            for ref, nombre, lat, km_paso in pasos:
                tramos.append([i, j, ref, nombre, lat, km_paso])
            time.sleep(0.3)
        print(f"  {ciudades[i]['ciudad']} listo")

    with open(CARPETA / "distancia_km.csv", "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows([[round(x, 3) for x in fila] for fila in dist])
    with open(CARPETA / "tiempo_h.csv", "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows([[round(x, 4) for x in fila] for fila in tiempo])
    with open(CARPETA / "km_por_via.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["i", "j", "ref", "nombre", "lat", "km"])
        w.writerows(tramos)

    print("3) Contorno de Espana")
    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/ESP.geo.json"
    (CARPETA / "espana.geojson").write_text(json.dumps(pedir_json(url)))

    (CARPETA / "fecha_descarga.txt").write_text(
        f"Datos descargados el {hoy}\n"
        "Coordenadas: Nominatim (OpenStreetMap)\n"
        "Rutas: OSRM, router.project-osrm.org, perfil driving\n"
        "Contorno: github.com/johan/world.geo.json (Natural Earth)\n"
    )
    print("Listo.")


if __name__ == "__main__":
    main()
