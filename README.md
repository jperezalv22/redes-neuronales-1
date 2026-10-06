# Trabajo 1: Optimización numérica, metaheurística y combinatoria

Redes neuronales y algoritmos bioinspirados, Universidad Nacional de Colombia, sede Medellín, 2026-02.

- **Parte 1.** Comparamos descenso por gradiente con un algoritmo genético, PSO y evolución diferencial sobre las funciones de **Rosenbrock** y **Rastrigin**, en 2D y 3D.
- **Parte 2.** Buscamos el recorrido más barato por las 47 capitales de provincia de la España peninsular con **colonia de hormigas** y un **algoritmo genético**. El costo de cada trayecto incluye el tiempo del vendedor, los peajes y el combustible.

El reporte (entrada de blog) está en [`reporte/blog.md`](reporte/blog.md).

## Instalación

Se necesita Python 3.10 o más reciente. Lo probamos con Python 3.14.

```bash
git clone <url-del-repositorio>
cd redes-neuronales

python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Las versiones de las librerías están fijadas en `requirements.txt`.

## Cómo correr todo

Cada experimento se lanza con un solo comando. Los resultados (tablas, figuras y GIF) quedan en `resultados/`.

| Qué | Comando | Tiempo aprox. |
|---|---|---|
| Verificar funciones y gradientes | `python pruebas.py` | 1 s |
| Parte 1: 30 corridas, tabla y gráficas | `python parte1_experimentos.py` | 1.5 min |
| Parte 1: demostración rápida | `python parte1_experimentos.py --corridas 5` | 20 s |
| Parte 1: GIF (GD y PSO) | `python parte1_animacion.py` | 25 s |
| Parte 2: ruta con ACO y GA + GIF | `python parte2_tsp.py --gif` | 30 s |
| Parte 2: estudio del valor de la hora | `python parte2_valor_hora.py` | 1 min |

Para ver todas las opciones de un script: `python <script>.py --help`.

### Cambiar parámetros sin tocar el código

Todo se cambia desde la línea de comandos. Algunos ejemplos:

```bash
# tasa de aprendizaje del descenso por gradiente y presupuesto de evaluaciones
python parte1_experimentos.py --eta 5e-4 --presupuesto 20000

# solo Rastrigin en 3D, con PSO de inercia más baja
python parte1_experimentos.py --funciones rastrigin --dims 3 --metodos pso --w 0.5

# GIF de evolución diferencial en Rosenbrock
python parte1_animacion.py --funcion rosenbrock --metodo de

# TSP: más evaporación y otro valor de la hora
python parte2_tsp.py --metodo aco --rho 0.3 --valor-hora 50

# TSP: genético sin búsqueda local 2-opt
python parte2_tsp.py --metodo ga --sin-2opt --generaciones 1000
```

Los datos del vehículo (consumo, precio del combustible) y el valor de la hora por defecto están en [`configuracion_tsp.json`](configuracion_tsp.json). También se pueden pasar con `--consumo`, `--precio` y `--valor-hora`.

## Semillas

Todo es reproducible.

- Parte 1: la corrida `k` usa la semilla `2026 + k`. Todos los métodos usan la misma semilla en la misma corrida. Se cambia con `--semilla`.
- GIF de la parte 1: semilla `2`.
- Parte 2: semilla `2026` para ACO y GA.

## Estructura

```
optim/                       Parte 1
  funciones.py               Rosenbrock, Rastrigin, gradientes y contador de evaluaciones
  gradiente.py               descenso por gradiente
  evolutivo.py               algoritmo genético con codificación real
  pso.py                     enjambre de partículas
  evolucion_diferencial.py   DE/rand/1/bin
tsp/                         Parte 2
  datos.py                   carga de datos y matriz de costos
  aco.py                     colonia de hormigas (Ant System)
  genetico.py                algoritmo genético con permutaciones (OX + inversión)
  utilidades.py              costo de un recorrido y 2-opt
  mapa.py                    mapa de España y GIF
datos/                       datos del TSP (no hace falta internet)
resultados/                  todo lo que generan los scripts
reporte/blog.md              reporte técnico
```

## Datos del TSP

Todos los datos están guardados en `datos/`, así que **ningún experimento necesita internet**. Solo `descargar_datos_tsp.py` se conecta, y no hace falta volverlo a correr.

| Archivo | Contenido | Fuente | Consulta |
|---|---|---|---|
| `capitales.csv` | nombre, provincia, latitud, longitud | Nominatim (OpenStreetMap) | 2026-10-06 |
| `distancia_km.csv` | matriz 47×47 de distancias por carretera | OSRM (router.project-osrm.org) | 2026-10-06 |
| `tiempo_h.csv` | matriz 47×47 de tiempos de viaje | OSRM | 2026-10-06 |
| `km_por_via.csv` | km que recorre cada ruta por cada vía (AP-6, A-1, ...) | OSRM | 2026-10-06 |
| `peajes_tarifas.csv` | autopistas de peaje y tarifa | Ministerio de Transportes, peajes vigentes desde el 01/01/2026 | 2026-10-06 |
| `espana.geojson` | contorno de España para el mapa | johan/world.geo.json (Natural Earth) | 2026-10-06 |
| `../configuracion_tsp.json` | vehículo, consumo, precio del gasóleo | km77.com y Boletín Petrolero de la UE | 2026-10-06 |

### Cómo se calcula el costo de un trayecto

```
costo(i, j) = valor_hora * tiempo(i, j)
            + peaje(i, j)
            + distancia(i, j) * consumo / 100 * precio_litro
```

Los **peajes** se calculan así: OSRM devuelve la ruta paso a paso con el código de cada vía. Sumamos los km que van por autopistas que siguen siendo de pago en 2026 y los multiplicamos por una tarifa en €/km. La tarifa sale de dividir el precio de un tramo completo (de las tablas oficiales del Ministerio) entre su longitud. Para las autopistas de las que no tenemos ese dato usamos el promedio. Es una aproximación y tiene limitaciones:

- La longitud de cada tramo de referencia es aproximada.
- Varias autopistas (AP-1, AP-2, AP-4, la AP-7 al norte de Alicante...) dejaron de cobrar peaje entre 2018 y 2026. La AP-7 y la AP-1 son de pago solo en algunos tramos, y los separamos por latitud (ver `tsp/datos.py`).
- OSRM elige la ruta más rápida y no tiene en cuenta el peaje. No buscamos rutas alternativas sin peaje.
- Las autopistas forales (AP-8, AP-1 en Gipuzkoa, AP-15 en Navarra) no aparecen en las tablas del Ministerio, así que su tarifa está por verificar.

Antes de entregar, **revisen los valores marcados como "VERIFICAR"** en `configuracion_tsp.json` y `datos/peajes_tarifas.csv`.

## Para volver a descargar los datos (opcional)

```bash
python descargar_datos_tsp.py     # unos 15 minutos, respeta los límites de Nominatim y OSRM
```

Si el servidor público de OSRM cambió su red de carreteras, las distancias pueden variar un poco.
