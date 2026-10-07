"""Dibujo del recorrido sobre el mapa de Espana y GIF de la evolucion."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .datos import CARPETA


def dibujar_contorno(ax):
    """Dibuja Espana (y en gris los paises vecinos) desde datos/espana.geojson."""
    geo = json.loads((CARPETA / "espana.geojson").read_text())
    ax.set_facecolor("#dbe9f4")  # mar
    for feature in geo["features"]:
        relleno = "#f2efe6" if feature["properties"]["iso"] == "ESP" else "#e0e0e0"
        g = feature["geometry"]
        poligonos = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poli in poligonos:
            anillo = np.array(poli[0])  # el primer anillo es el borde exterior
            ax.fill(anillo[:, 0], anillo[:, 1], color=relleno, ec="#999999", lw=0.6)
    ax.set_xlim(-9.6, 3.6)
    ax.set_ylim(35.8, 44.0)
    # corregimos la deformacion: 1 grado de longitud mide menos que 1 de latitud
    ax.set_aspect(1 / np.cos(np.radians(40)))
    ax.set_xticks([])
    ax.set_yticks([])


def dibujar_ruta(ax, datos, ruta, color="#c0392b"):
    r = list(ruta) + [ruta[0]]  # cerrar el recorrido
    linea, = ax.plot(datos.lon[r], datos.lat[r], "-", color=color, lw=1.6)
    return linea


def figura_ruta(datos, ruta, titulo, archivo):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    dibujar_contorno(ax)
    dibujar_ruta(ax, datos, ruta)
    ax.plot(datos.lon, datos.lat, "o", color="#2c3e50", ms=4)
    ax.plot(datos.lon[ruta[0]], datos.lat[ruta[0]], "s", color="#27ae60", ms=8, label="salida")
    for k, nombre in enumerate(datos.nombres):
        ax.annotate(nombre, (datos.lon[k], datos.lat[k]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_title(titulo)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(archivo, dpi=130)
    plt.close(fig)


def gif_evolucion(datos, historia, titulo, archivo, fps=4):
    """Un cuadro por cada vez que mejoro la mejor solucion."""
    fig, ax = plt.subplots(figsize=(7, 5.8))
    dibujar_contorno(ax)
    ax.plot(datos.lon, datos.lat, "o", color="#2c3e50", ms=4, zorder=3)
    linea, = ax.plot([], [], "-", color="#c0392b", lw=1.6)

    def cuadro(k):
        it, costo, ruta = historia[k]
        r = list(ruta) + [ruta[0]]
        linea.set_data(datos.lon[r], datos.lat[r])
        ax.set_title(f"{titulo}\niteración {it}, costo = {costo:,.2f} €")
        return (linea,)

    # repetimos el ultimo cuadro para que el GIF "descanse" al final
    cuadros = list(range(len(historia))) + [len(historia) - 1] * 6
    anim = FuncAnimation(fig, cuadro, frames=cuadros, blit=False)
    anim.save(archivo, writer=PillowWriter(fps=fps))
    plt.close(fig)
