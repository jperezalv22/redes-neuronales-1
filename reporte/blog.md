---
title: "Gradientes, enjambres y hormigas: de Rosenbrock a recorrer España en carro"
curso: Redes neuronales y algoritmos bioinspirados, UNAL 2026-02
autores: "<nombres del equipo>"
repositorio: "<enlace al repositorio Git>"
---

# Gradientes, enjambres y hormigas: de Rosenbrock a recorrer España en carro

**Repositorio:** <enlace al repositorio Git>

En este trabajo comparamos métodos de optimización que usan el gradiente con métodos metaheurísticos que solo evalúan la función. Primero lo hacemos con dos funciones de prueba clásicas, Rosenbrock y Rastrigin. Después pasamos a un problema combinatorio real: el recorrido más barato en carro por las 47 capitales de provincia de la España peninsular.

Todo el código, los datos y las instrucciones para reproducir cada figura están en el repositorio. Cada figura y cada tabla de este reporte se regenera con un solo comando (ver el README).

---

## Parte 1. Optimización numérica

### 1.1 Funciones de prueba

Escogimos dos funciones con geometrías opuestas, las dos definidas para cualquier dimensión $n$ (las de Goldstein-Price y las seis jorobas de camello solo existen en 2D, y necesitábamos 2D y 3D).

**Rosenbrock** (Rosenbrock, 1960):

$$f(\mathbf{x}) = \sum_{i=1}^{n-1}\left[100\,(x_{i+1}-x_i^2)^2 + (1-x_i)^2\right], \qquad \mathbf{x}\in[-2.048,\,2.048]^n$$

Mínimo global: $f(1,\dots,1)=0$. Tiene un solo mínimo, pero está al fondo de un valle largo, curvo y muy estrecho. En 2D la hessiana en el mínimo es

$$H(1,1)=\begin{pmatrix}802 & -400\\ -400 & 200\end{pmatrix},$$

con valores propios $\lambda_{\max}\approx 1001.6$ y $\lambda_{\min}\approx 0.399$. El número de condición es $\kappa\approx 2508$: la función es casi 2500 veces más empinada en una dirección que en la otra.

**Rastrigin** (Rastrigin, 1974; ver también Jamil y Yang, 2013):

$$f(\mathbf{x}) = 10n + \sum_{i=1}^{n}\left[x_i^2 - 10\cos(2\pi x_i)\right], \qquad \mathbf{x}\in[-5.12,\,5.12]^n$$

Mínimo global: $f(0,\dots,0)=0$. Es una parábola con una "rejilla" de cosenos encima, así que tiene un mínimo local cerca de cada punto de coordenadas enteras: unos $11^n$ mínimos locales dentro del dominio (121 en 2D y 1331 en 3D). Cada mínimo local es convexo localmente, pero la función en conjunto no lo es.

Los gradientes son

$$\frac{\partial f_{\text{Ros}}}{\partial x_i} = -400\,x_i(x_{i+1}-x_i^2) - 2(1-x_i) + 200(x_i - x_{i-1}^2),$$

donde el primer y segundo término solo existen si $i<n$ y el tercero solo si $i>1$, y

$$\frac{\partial f_{\text{Ras}}}{\partial x_i} = 2x_i + 20\pi\sin(2\pi x_i).$$

Antes de usarlos los comparamos con diferencias centrales en 100 puntos aleatorios para $n=2,3,5$ (`python pruebas.py`). El error relativo máximo fue del orden de $10^{-10}$.

### 1.2 Métodos

**Descenso por gradiente (GD).** Con tasa de aprendizaje fija $\eta$:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\nabla f(\mathbf{x}_k).$$

Usamos $\eta=10^{-3}$ y un punto inicial uniforme en el dominio. Se detiene si $\lVert\nabla f\rVert<10^{-8}$, si se acaba el presupuesto o si $f$ deja de ser finito (divergencia). Para una cuadrática, GD es estable solo si $\eta<2/\lambda_{\max}$ (Nocedal y Wright, 2006). Cerca del mínimo de Rosenbrock eso da $\eta<0.002$.

**Algoritmo genético (GA)** con codificación real (Holland, 1975; Goldberg, 1989). En cada generación: selección por torneo de tamaño 3, cruce BLX-$\alpha$ con $\alpha=0.5$ y probabilidad 0.9, y mutación gaussiana por gen con probabilidad 0.1 y $\sigma = 0.1\times$(ancho del dominio). Con elitismo de 1 individuo y población de 30. En el cruce BLX-$\alpha$ cada gen del hijo es

$$h_j \sim U\big[\min(p_{1j},p_{2j}) - \alpha d_j,\ \max(p_{1j},p_{2j}) + \alpha d_j\big],\quad d_j=|p_{1j}-p_{2j}|.$$

**Optimización por enjambre de partículas (PSO)** (Kennedy y Eberhart, 1995) con peso de inercia:

$$\mathbf{v}_i \leftarrow w\,\mathbf{v}_i + c_1\mathbf{r}_1\odot(\mathbf{p}_i-\mathbf{x}_i) + c_2\mathbf{r}_2\odot(\mathbf{g}-\mathbf{x}_i),\qquad \mathbf{x}_i\leftarrow\mathbf{x}_i+\mathbf{v}_i,$$

donde $\mathbf{p}_i$ es la mejor posición de la partícula, $\mathbf{g}$ la mejor del enjambre y $\mathbf{r}_1,\mathbf{r}_2\sim U[0,1]^n$. Usamos $w=0.7298$ y $c_1=c_2=1.49618$ (Clerc y Kennedy, 2002), 30 partículas y velocidad máxima del 20 % del ancho del dominio.

**Evolución diferencial (DE)**, variante DE/rand/1/bin (Storn y Price, 1997). Para cada individuo $\mathbf{x}_i$:

$$\mathbf{v} = \mathbf{x}_{r_1} + F(\mathbf{x}_{r_2}-\mathbf{x}_{r_3}),\qquad u_j = \begin{cases} v_j & \text{si } U_j<CR \text{ o } j=j_{\text{rand}}\\ x_{ij} & \text{si no}\end{cases}$$

y $\mathbf{u}$ reemplaza a $\mathbf{x}_i$ si $f(\mathbf{u})\le f(\mathbf{x}_i)$. Usamos $F=0.8$, $CR=0.9$ y población de 30.

En los tres métodos de población, si un punto se sale del dominio lo devolvemos al borde.

### 1.3 Comparación justa: cómo contamos las evaluaciones

Cada método llama a la función a través de una clase `Problema` (en `optim/funciones.py`) que cuenta por separado las evaluaciones de $f$ ($n_f$) y las de $\nabla f$ ($n_\nabla$).

**Equivalencia propuesta:** una evaluación de $\nabla f$ cuesta $2n$ evaluaciones de $f$, que es lo que cuesta aproximar el gradiente con diferencias centrales:

$$\frac{\partial f}{\partial x_i}\approx\frac{f(\mathbf{x}+h\mathbf{e}_i)-f(\mathbf{x}-h\mathbf{e}_i)}{2h}.$$

Elegimos esta equivalencia porque es lo que pagaría alguien que no tiene la fórmula del gradiente, y porque es fácil de verificar en el código. Hay un argumento a favor de un costo menor: con diferenciación automática en modo reverso el gradiente cuesta un número pequeño y constante de evaluaciones de $f$, sin importar $n$ (Griewank y Walther, 2008). En 2D y 3D las dos opciones dan números parecidos ($2n$ = 4 o 6). En todo caso, en la Tabla 1 reportamos $n_f$ y $n_\nabla$ por separado, así que el lector puede usar otra equivalencia.

El costo de una corrida es entonces

$$\text{costo} = n_f + 2n\cdot n_\nabla .$$

Todos los métodos tienen el mismo presupuesto: **10 000 evaluaciones equivalentes**. GD evalúa además $f$ en cada iteración para poder dibujar la curva de convergencia, y esas evaluaciones también se cuentan, así que cada iteración le cuesta $2n+1$.

**Protocolo.** 30 corridas independientes por configuración (función × dimensión × método). La corrida $k$ usa la semilla $2026+k$, la misma para todos los métodos. Definimos **éxito** como $f(\mathbf{x}_{\text{final}}) - f^* < 10^{-3}$.

### 1.4 Resultados

<!-- tabla generada por: python parte1_experimentos.py  (resultados/parte1/tabla_resultados.md) -->

**Tabla 1.** Valor final $f-f^*$ en 30 corridas por configuración, con presupuesto de 10 000 evaluaciones equivalentes.

| Función | Dim | Método | Media | Desv. est. | Mejor | Peor | Éxito (%) | Evals $f$ | Evals $\nabla f$ |
|---|---|---|---|---|---|---|---|---|---|
| Rosenbrock | 2 | Descenso por gradiente | 3.69e-02 | 2.15e-02 | 7.81e-04 | 9.87e-02 | 3 | 2000 | 1999 |
| Rosenbrock | 2 | Algoritmo genético | 3.22e-02 | 5.18e-02 | 4.24e-07 | 2.28e-01 | 23 | 9977 | 0 |
| Rosenbrock | 2 | PSO | 8.28e-12 | 2.47e-11 | 3.67e-17 | 1.12e-10 | 100 | 9990 | 0 |
| Rosenbrock | 2 | Evolución diferencial | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 100 | 9990 | 0 |
| Rosenbrock | 3 | Descenso por gradiente | 2.18e+02 | 8.37e+02 | 7.20e-05 | 3.73e+03 | 7 | 1334 | 1333 |
| Rosenbrock | 3 | Algoritmo genético | 7.69e-01 | 9.08e-01 | 1.22e-02 | 3.49e+00 | 0 | 9977 | 0 |
| Rosenbrock | 3 | PSO | 3.55e-03 | 2.93e-03 | 3.93e-05 | 1.28e-02 | 17 | 9990 | 0 |
| Rosenbrock | 3 | Evolución diferencial | 6.65e-23 | 2.02e-22 | 2.08e-26 | 1.10e-21 | 100 | 9990 | 0 |
| Rastrigin | 2 | Descenso por gradiente | 1.54e+01 | 1.12e+01 | 0.00e+00 | 3.38e+01 | 7 | 50 | 50 |
| Rastrigin | 2 | Algoritmo genético | 1.34e-05 | 6.40e-05 | 0.00e+00 | 3.49e-04 | 100 | 9977 | 0 |
| Rastrigin | 2 | PSO | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 100 | 9990 | 0 |
| Rastrigin | 2 | Evolución diferencial | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 100 | 9990 | 0 |
| Rastrigin | 3 | Descenso por gradiente | 2.35e+01 | 1.50e+01 | 1.99e+00 | 5.87e+01 | 0 | 52 | 52 |
| Rastrigin | 3 | Algoritmo genético | 3.58e-06 | 1.10e-05 | 0.00e+00 | 5.75e-05 | 100 | 9977 | 0 |
| Rastrigin | 3 | PSO | 2.65e-01 | 5.80e-01 | 0.00e+00 | 1.99e+00 | 80 | 9990 | 0 |
| Rastrigin | 3 | Evolución diferencial | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 100 | 9990 | 0 |

*Fuente: elaboración propia con `python parte1_experimentos.py`. Los valores "0.00e+00" son ceros exactos en punto flotante. Las columnas de evaluaciones son promedios por corrida.*

![Figura 1](../resultados/parte1/convergencia.png)

**Figura 1.** Mediana (línea) y rango intercuartílico (sombra) del mejor valor encontrado contra el número de evaluaciones equivalentes. *Fuente: elaboración propia.*

![Figura 2](../resultados/parte1/cajas_valor_final.png)

**Figura 2.** Distribución del valor final en las 30 corridas (los ceros se dibujan como $10^{-16}$). *Fuente: elaboración propia.*

![Figura 3](../resultados/parte1/anim_rosenbrock_gd.gif)

**Figura 3.** Descenso por gradiente en Rosenbrock 2D, desde $(-0.98, -0.83)$ con $\eta=10^{-3}$. *Fuente: elaboración propia con `python parte1_animacion.py`.*

![Figura 4](../resultados/parte1/anim_rastrigin_gd.gif)

**Figura 4.** Descenso por gradiente en Rastrigin 2D. Se queda en el mínimo local más cercano al punto inicial. *Fuente: elaboración propia.*

![Figura 5](../resultados/parte1/anim_rastrigin_pso.gif)

**Figura 5.** PSO en Rastrigin 2D: el enjambre explora varios "huecos" y luego se concentra en el origen. *Fuente: elaboración propia.*

![Figura 6](../resultados/parte1/anim_rosenbrock_pso.gif)

**Figura 6.** PSO en Rosenbrock 2D. *Fuente: elaboración propia.*

### 1.5 Discusión

**Rosenbrock: un solo mínimo, pero mal condicionado.** Como se ve en la Figura 3, GD baja muy rápido hasta el fondo del valle en unas pocas iteraciones y después avanza muy despacio por él. La razón es el condicionamiento: el paso estable lo fija la dirección empinada ($\eta<2/\lambda_{\max}$), y con ese paso el avance en la dirección plana es de orden $\eta\lambda_{\min}\approx 4\times10^{-4}$ por iteración. Por eso, después de ~2000 gradientes en 2D, la media sigue en $3.7\times10^{-2}$ y solo 1 de 30 corridas cumple el umbral (Tabla 1). En 3D pasó algo peor: 2 de las 30 corridas divergieron en menos de 10 iteraciones porque empezaron en zonas donde $\lambda_{\max}$ es mucho mayor que cerca del mínimo, y el mismo $\eta$ ya no es estable. Esas dos corridas son las que inflan la media a 218. Así, el tamaño de paso de GD depende de la geometría y no hay uno bueno para todo el dominio.

Entre las metaheurísticas, DE fue la mejor: llegó al cero de máquina en 2D en ~3500 evaluaciones y en 3D en ~7000 (Figura 1). El vector diferencia $\mathbf{x}_{r_2}-\mathbf{x}_{r_3}$ se alinea solo con el valle, porque los individuos se reparten a lo largo de él, y eso le da pasos en la dirección correcta sin calcular ninguna derivada. PSO funcionó muy bien en 2D y mucho peor en 3D (17 % de éxito). El GA con BLX y mutación gaussiana fue el más débil en Rosenbrock: su mutación no tiene ninguna dirección preferida y el valle es muy estrecho.

**Rastrigin: muchos mínimos locales.** Aquí GD no tiene opción. Con $\eta=10^{-3}$ el paso máximo es $\eta\cdot\max|\partial f/\partial x_i|\approx 0.07$, que no alcanza para saltar de un hueco a otro, así que converge al mínimo local del hueco donde empezó en unas 50 iteraciones (Figura 4) y se detiene porque el gradiente se anula. Usó solo unas 250 evaluaciones equivalentes en 2D (370 en 3D) de las 10 000. Las 2 corridas exitosas en 2D fueron las que empezaron, por suerte, dentro de la cuenca del origen ($|x_i|<0.5$; puntos iniciales $(-0.135, 0.339)$ y $(0.034, -0.194)$). La probabilidad de que eso pase es de alrededor de $(1/10.24)^2\approx 1\%$ en 2D. Los tres métodos de población resolvieron Rastrigin 2D en el 100 % de las corridas. En 3D, PSO falló en 6 de 30 corridas, quedando en $f\approx 0.995$ o $1.99$, que son los mínimos locales vecinos al origen (una o dos coordenadas en $\pm1$). El GA, que sufrió en Rosenbrock, aquí fue de los mejores: su mutación grande le permite saltar entre huecos.

**¿Qué aportó cada familia?**

- El **gradiente** aporta información local muy barata: en Rastrigin llegó a un mínimo local con 50 gradientes, y en un problema convexo y bien condicionado sería imbatible. Pero no sabe nada de lo que pasa fuera de su cuenca, y su velocidad depende del número de condición.
- Las **metaheurísticas** aportan exploración global y no necesitan derivadas, a cambio de muchas más evaluaciones de $f$. Su desempeño depende del operador: DE se adapta a la escala y orientación del problema porque usa diferencias entre individuos; PSO converge rápido pero puede estancarse; el GA explora bien pero afina mal.
- Una estrategia razonable en la práctica es combinarlas: una metaheurística para encontrar la cuenca correcta y luego gradiente (o un método de segundo orden) para afinar. También se podría hacer GD con varios reinicios. Con el presupuesto que GD dejó sin usar en Rastrigin se podrían haber hecho unos 40 reinicios en 2D, aunque con un 1 % de probabilidad por reinicio de caer en la cuenca correcta seguiría siendo peor que PSO o DE.

---

## Parte 2. Optimización combinatoria: recorrer España

### 2.1 Formulación del problema del viajero

Tenemos $N=47$ ciudades (las capitales de provincia peninsulares) y un costo $C_{ij}\ge 0$ de ir de $i$ a $j$. Un recorrido cerrado es una permutación $\pi=(\pi_1,\dots,\pi_N)$ de las ciudades, y su costo es

$$L(\pi) = \sum_{k=1}^{N-1} C_{\pi_k\pi_{k+1}} + C_{\pi_N\pi_1}.$$

El problema es encontrar $\pi^* = \arg\min_\pi L(\pi)$. Como el recorrido es cerrado, da igual en qué ciudad empiece, y como nuestros costos son simétricos ($C_{ij}=C_{ji}$) también da igual el sentido. Quedan $(N-1)!/2\approx 2.7\times10^{57}$ recorridos distintos, así que no se pueden revisar todos.

**Costo de un trayecto.** Para cada par de ciudades,

$$C_{ij}(h) = h\,t_{ij} + p_{ij} + d_{ij}\,\frac{c}{100}\,P,$$

donde $h$ es el valor de la hora del vendedor (€/h), $t_{ij}$ el tiempo de viaje (h), $p_{ij}$ el peaje (€), $d_{ij}$ la distancia (km), $c$ el consumo del vehículo (l/100 km) y $P$ el precio del combustible (€/l).

### 2.2 Datos

La Tabla 2 resume de dónde sale cada dato. Todos están guardados en la carpeta `datos/` del repositorio, así que los experimentos no necesitan internet.

**Tabla 2.** Fuentes de los datos del TSP.

| Dato | Valor / archivo | Fuente | Fecha de consulta |
|---|---|---|---|
| Coordenadas de las 47 capitales | `capitales.csv` | Nominatim (OpenStreetMap contributors, 2026) | 2026-10-06 |
| Distancia y tiempo por carretera | `distancia_km.csv`, `tiempo_h.csv` | OSRM (Luxen y Vetter, 2011), servidor público, perfil de carro | 2026-10-06 |
| Km por cada vía en cada ruta | `km_por_via.csv` | OSRM | 2026-10-06 |
| Tarifas de peaje 2026 | `peajes_tarifas.csv` | Ministerio de Transportes y Movilidad Sostenible (2026) | 2026-10-06 |
| Vehículo | SEAT León 2.0 TDI 85 kW (115 CV), diésel, **4.6 l/100 km** (WLTP combinado) | km77.com (2024) | 2026-10-06 |
| Precio del gasóleo | **1.934 €/l** (gasóleo de automoción con impuestos, España) | Comisión Europea (2026), Weekly Oil Bulletin, precios al 28/09/2026 | 2026-10-06 |
| Valor de la hora | 15 €/h por defecto, estudiado entre 0 y 200 €/h | parámetro | — |

*Fuente: elaboración propia.*

**Peajes.** Esta fue la parte más difícil. Entre 2018 y 2021 se liberaron muchas autopistas (la AP-1 Burgos–Armiñón, la AP-4, la AP-7 al norte de Alicante, la AP-2...) y en 2026 se liberó la circunvalación de Alicante. Lo que hicimos fue:

1. Pedir a OSRM la ruta de cada uno de los 1081 pares de ciudades, paso a paso. Cada paso trae el código de la vía (por ejemplo `AP-68`) y su longitud.
2. Sumar los km que van por autopistas que siguen siendo de pago en 2026 según la lista del Ministerio. La AP-7 solo cuenta al sur de Alicante y la AP-1 solo entre Vitoria y Eibar.
3. Multiplicar por una tarifa en €/km. Para la AP-68, AP-6, AP-71 y AP-53 la calculamos con el importe oficial del tramo completo para vehículos ligeros dividido entre su longitud (0.136, 0.224, 0.163 y 0.129 €/km). Para las demás usamos el promedio, 0.163 €/km.

Como comprobación, el modelo da 39.79 € para Bilbao–Zaragoza (293 km por la AP-68), y la tabla oficial dice 39.90 €. Con este método, 561 de los 1081 trayectos pagan algo de peaje.

Las limitaciones están documentadas en el README: las longitudes de los tramos son aproximadas, OSRM siempre da la ruta más rápida sin mirar el peaje (no exploramos rutas alternativas sin peaje para cada par) y las tarifas de las autopistas forales del País Vasco y Navarra están por verificar. En el mapa dibujamos cada trayecto como una línea recta entre ciudades, aunque el costo usa la ruta real por carretera.

### 2.3 Métodos

**Colonia de hormigas (Ant System)** (Dorigo et al., 1996; Dorigo y Stützle, 2004). En cada iteración, $m$ hormigas construyen un recorrido. Estando en $i$, la hormiga $k$ va a una ciudad $j$ no visitada con probabilidad

$$p_{ij}^k = \frac{\tau_{ij}^{\alpha}\,\eta_{ij}^{\beta}}{\sum_{l\,\text{no visitada}} \tau_{il}^{\alpha}\,\eta_{il}^{\beta}},\qquad \eta_{ij}=\frac{1}{C_{ij}},$$

donde $\tau_{ij}$ es la feromona de la arista. Al final de la iteración la feromona se evapora y cada hormiga deposita una cantidad inversamente proporcional al costo de su recorrido $L_k$:

$$\tau_{ij} \leftarrow (1-\rho)\,\tau_{ij} + \sum_{k=1}^{m}\Delta\tau_{ij}^k,\qquad \Delta\tau_{ij}^k=\begin{cases}Q/L_k & \text{si } (i,j)\text{ está en el recorrido de } k\\ 0 & \text{si no.}\end{cases}$$

Usamos $m=30$, $\alpha=1$, $\beta=3$, $\rho=0.1$, $Q=1$, 150 iteraciones y feromona inicial $\tau_0=m/L_{nn}$, con $L_{nn}$ el costo del recorrido del vecino más cercano (Dorigo y Stützle, 2004).

**Algoritmo genético con permutaciones.** Cada individuo es un recorrido. Usamos selección por torneo de tamaño 5, cruce OX (*order crossover*; Davis, 1985) con probabilidad 0.9, mutación por inversión de un segmento con probabilidad 0.3, elitismo de 2 individuos, población de 100 y 300 generaciones. El cruce OX copia un segmento del primer padre y completa el resto de ciudades en el orden en que aparecen en el segundo, así que el hijo siempre es una permutación válida.

**Búsqueda local 2-opt** (Croes, 1958). Quita dos aristas $(a,b)$ y $(c,d)$ y reconecta como $(a,c)$ y $(b,d)$, que equivale a invertir el tramo entre $b$ y $c$. Acepta el cambio si

$$\Delta = C_{ac}+C_{bd}-C_{ab}-C_{cd} < 0,$$

y repite hasta que no haya mejora. En la colonia de hormigas aplicamos 2-opt al mejor recorrido de cada iteración. En el genético la aplicamos a cada hijo con probabilidad 0.05 (ver el Hallazgo 4 en la cacería de la alucinación para saber por qué así).

### 2.4 Resultados

Con $h=15$ €/h, el mejor recorrido que encontramos cuesta **1 792.51 €** (Figura 7). Lo encontró el algoritmo genético con la semilla 2026. La colonia de hormigas, con la misma semilla, se quedó en 1 803.25 € (0.6 % más caro), pero con otras semillas también llega a 1 792.51 € (Tabla 4). La Tabla 3 muestra el desglose del mejor recorrido.

**Tabla 3.** Desglose del mejor recorrido encontrado con $h=15$ €/h.

| Concepto | Valor |
|---|---|
| Distancia total | 6 335.4 km |
| Tiempo de conducción | 79.52 h |
| Costo del tiempo | 1 192.80 € |
| Peajes | 36.08 € |
| Combustible | 563.62 € |
| **Costo total** | **1 792.51 €** |

*Fuente: elaboración propia con `python parte2_tsp.py`.*

Orden: Bilbao → Santander → Oviedo → A Coruña → Lugo → Pontevedra → Ourense → Zamora → León → Burgos → Palencia → Valladolid → Segovia → Ávila → Salamanca → Cáceres → Badajoz → Huelva → Sevilla → Cádiz → Córdoba → Jaén → Granada → Málaga → Almería → Murcia → Alacant → Albacete → Ciudad Real → Toledo → Madrid → Guadalajara → Cuenca → Teruel → València → Castelló de la Plana → Tarragona → Barcelona → Girona → Lleida → Huesca → Zaragoza → Soria → Logroño → Pamplona → Donostia-San Sebastián → Vitoria-Gasteiz → (Bilbao). Como el recorrido es cerrado, se puede empezar en cualquier ciudad.

![Figura 7](../resultados/parte2/ruta_ga.png)

**Figura 7.** Mejor recorrido encontrado con $h=15$ €/h (algoritmo genético con 2-opt). *Fuente: elaboración propia; contorno de Natural Earth (s. f.).*

![Figura 8](../resultados/parte2/sin_2opt/evolucion_aco.gif)

**Figura 8.** Evolución de la mejor solución de la colonia de hormigas sin 2-opt: cada cuadro es una mejora. *Fuente: elaboración propia con `python parte2_tsp.py --sin-2opt --gif`.*

![Figura 9](../resultados/parte2/sin_2opt/evolucion_ga.gif)

**Figura 9.** Evolución de la mejor solución del algoritmo genético sin 2-opt. *Fuente: elaboración propia.*

**¿Es el óptimo?** No podemos demostrarlo, porque no resolvimos el problema con un método exacto. Lo que sí hicimos fue correr cada configuración 10 veces con semillas distintas (Tabla 4). El recorrido de 1 792.51 € apareció 4 de 10 veces con ACO + 2-opt y 7 de 10 con GA + 2-opt, y ninguna corrida encontró algo mejor. Eso nos da bastante confianza de que es el óptimo o está muy cerca.

**Tabla 4.** Costo final en 10 corridas por configuración ($h=15$ €/h). La brecha se mide contra el mejor costo encontrado (1 792.51 €).

| Configuración | Media (€) | Desv. (€) | Mejor (€) | Peor (€) | Brecha media | Veces que llega al mejor | s/corrida |
|---|---|---|---|---|---|---|---|
| ACO + 2-opt | 1 795.41 | 4.16 | 1 792.51 | 1 803.25 | 0.16 % | 4/10 | 3.2 |
| ACO sin 2-opt | 1 876.73 | 18.92 | 1 841.46 | 1 899.88 | 4.70 % | 0/10 | 2.9 |
| GA + 2-opt | 1 794.15 | 2.64 | 1 792.51 | 1 797.96 | 0.09 % | 7/10 | 4.0 |
| GA sin 2-opt | 1 985.77 | 71.55 | 1 862.48 | 2 077.23 | 10.78 % | 0/10 | 1.4 |

*Fuente: elaboración propia con `python parte2_comparacion.py`.*

Sin 2-opt, la colonia de hormigas queda en promedio a 4.7 % del mejor y el genético a 10.8 %. La colonia de hormigas por sí sola es bastante mejor que el genético: su regla de construcción usa el costo de cada arista ($\eta_{ij}$), mientras que el cruce OX y la inversión no saben nada de costos y solo los "ven" a través de la selección. Con 2-opt los dos quedan a menos de 0.2 % en promedio. En la práctica, la búsqueda local hace la mayor parte del trabajo y la metaheurística le da buenos puntos de partida.

### 2.5 ¿Cambia la ruta con el valor de la hora?

Para 15 valores de $h$ entre 0 y 200 €/h corrimos la colonia de hormigas y el genético (los dos con 2-opt). Para no confundir el ruido de los algoritmos con un cambio real, al final evaluamos todos los recorridos encontrados (30 en total) con el costo de cada $h$ y nos quedamos con el más barato.

Para un recorrido fijo $\pi$, el costo es una recta en $h$:

$$L_\pi(h) = \underbrace{\textstyle\sum (p_{ij} + d_{ij}cP/100)}_{\text{dinero } M_\pi} + h\,\underbrace{\textstyle\sum t_{ij}}_{\text{horas } T_\pi}.$$

El óptimo para cada $h$ es la recta más baja, así que la ruta óptima solo cambia donde se cruzan dos rectas, en $h^* = (M_B - M_A)/(T_A - T_B)$. Valores bajos de $h$ premian rutas baratas en dinero (menos peajes) y valores altos premian rutas rápidas.

**Tabla 5.** Mejor recorrido encontrado para cada valor de la hora. "Aristas distintas" cuenta cuántos trayectos cambian respecto al recorrido con $h=0$.

| Valor hora (€/h) | km | horas | peajes (€) | combustible (€) | costo total (€) | aristas distintas |
|---|---|---|---|---|---|---|
| 0 | 6 379.3 | 80.75 | 22.79 | 567.53 | 590.32 | 0 |
| 2.5 | 6 379.3 | 80.75 | 22.79 | 567.53 | 792.19 | 0 |
| 5 | 6 379.3 | 80.75 | 22.79 | 567.53 | 994.05 | 0 |
| 7.5 | 6 416.6 | 79.85 | 26.01 | 570.85 | 1 195.73 | 10 |
| 10 | 6 335.4 | 79.52 | 36.08 | 563.62 | 1 394.91 | 10 |
| 15 | 6 335.4 | 79.52 | 36.08 | 563.62 | 1 792.51 | 10 |
| 20 | 6 242.3 | 78.32 | 67.66 | 555.34 | 2 189.49 | 15 |
| 25 | 6 242.3 | 78.32 | 67.66 | 555.34 | 2 581.11 | 15 |
| 30 | 6 247.0 | 78.05 | 74.97 | 555.76 | 2 972.24 | 15 |
| 40 | 6 247.0 | 78.05 | 74.97 | 555.76 | 3 752.74 | 15 |
| 50 | 6 247.0 | 78.05 | 74.97 | 555.76 | 4 533.25 | 15 |
| 75 | 6 247.0 | 78.05 | 74.97 | 555.76 | 6 484.51 | 15 |
| 100 | 6 247.0 | 78.05 | 74.97 | 555.76 | 8 435.77 | 15 |
| 150 | 6 205.3 | 77.96 | 90.45 | 552.05 | 12 336.12 | 18 |
| 200 | 6 205.3 | 77.96 | 90.45 | 552.05 | 16 234.00 | 18 |

*Fuente: elaboración propia con `python parte2_valor_hora.py`.*

La Tabla 6 da los valores exactos de $h$ en los que cambia la ruta, calculados con los cruces de las rectas de recorridos consecutivos.

**Tabla 6.** Valores de la hora en los que cambia el mejor recorrido encontrado.

| Cambio | $h^*$ (€/h) | Horas ahorradas | Dinero extra (€) |
|---|---|---|---|
| 1.º | 7.30 | 0.90 | 6.54 |
| 2.º | 8.65 | 0.33 | 2.85 |
| 3.º | 19.48 | 1.20 | 23.29 |
| 4.º | 28.20 | 0.27 | 7.73 |
| 5.º | 126.68 | 0.09 | 11.77 |

*Fuente: elaboración propia con `python parte2_valor_hora.py`.*

![Figura 10](../resultados/parte2/valor_hora.png)

**Figura 10.** Distancia, tiempo y peajes del mejor recorrido según el valor de la hora. *Fuente: elaboración propia.*

![Figura 11](../resultados/parte2/rutas_valor_hora.png)

**Figura 11.** Mejor recorrido con $h=0$ (solo dinero) y con $h=200$ €/h (casi solo tiempo). *Fuente: elaboración propia.*

Así, **sí cambia la ruta**, y el primer cambio ocurre en $h^*\approx 7.3$ €/h. Por debajo de ese valor al vendedor le conviene el recorrido que paga solo 22.79 € de peajes. El cambio más grande ocurre en $h^*\approx 19.5$ €/h: se pagan 23.29 € más (sobre todo la AP-9 entre A Coruña y Pontevedra y la AP-7 entre Cádiz y Málaga) para ahorrar 1.2 h. Por encima de unos 30 €/h los cambios ya son muy pequeños (0.27 h y 0.09 h).

Como se ve en la Tabla 5, los cambios son pequeños en términos relativos: entre $h=0$ y $h=200$ el tiempo total baja solo 2.8 h (de 80.75 a 77.96 h), y los peajes pasan de 23 a 90 €. Esto tiene sentido porque los costos de tiempo y de combustible son casi proporcionales entre sí (los dos crecen con la distancia). Lo que realmente separa una ruta "barata" de una "rápida" son los pocos trayectos con autopistas de pago, que en 2026 son muchos menos que hace unos años. Comparando los dos recorridos de la Figura 11, 18 de los 47 trayectos cambian. Los que entran con $h=200$ y pagan peaje son A Coruña–Pontevedra (AP-9), León–Oviedo (AP-66), Lugo–León (AP-71), Madrid–Ávila (AP-6 y AP-51), Cádiz–Málaga (AP-7 de la Costa del Sol) y dos trayectos en el País Vasco. Justamente son las zonas donde siguen las autopistas de pago.

Hay que tener cuidado con los cambios pequeños. Los valores de $h^*$ los calculamos entre recorridos que encontraron los algoritmos, no entre todos los posibles. Una primera versión del estudio usaba solo la colonia de hormigas. Esa versión también encontró el primer cambio en 7.30 €/h, pero ubicó el cambio grande en 17.14 €/h, porque no había encontrado el recorrido de 79.52 h que sí encontró el genético. Para $h$ alto los recorridos también variaron un poco entre versiones (menos de 0.1 h de diferencia), así que el último cambio (126.68 €/h) puede deberse a que los algoritmos no encontraron antes ese recorrido. Lo que se mantiene es el patrón: un primer cambio cerca de 7 €/h, uno grande entre 17 y 20 €/h y cambios muy pequeños de ahí en adelante.

---

## Conclusiones

- En funciones continuas, el gradiente es muy eficiente cuando el problema es localmente convexo y bien condicionado, pero se estanca en el primer mínimo local (Rastrigin) y avanza muy despacio en valles mal condicionados (Rosenbrock). Su tasa de aprendizaje depende de la curvatura, y una tasa fija puede divergir en una parte del dominio y ser lenta en otra.
- Con el mismo presupuesto de evaluaciones, evolución diferencial fue el método más confiable en las cuatro configuraciones. PSO fue rápido pero se estancó en algunos casos de 3D, y el GA de codificación real fue bueno en Rastrigin y malo en Rosenbrock.
- En el TSP, la colonia de hormigas y el genético llegan al mismo recorrido (1 792.51 € con $h=15$ €/h) cuando se combinan con 2-opt. Sin búsqueda local, la colonia de hormigas es claramente mejor.
- La ruta óptima depende del valor de la hora. El primer cambio ocurre cerca de 7.3 €/h y el más importante cerca de 19.5 €/h. Por encima de unos 30 €/h la ruta casi no cambia.
- Obtener los datos fue tan difícil como optimizar: los peajes cambian cada año, las etiquetas de OpenStreetMap no siempre son consistentes y un resumen automático nos dio tarifas equivocadas.

---

## Uso de inteligencia artificial

<!--
COMPLETAR POR EL EQUIPO. El enunciado pide los principales prompts (literales)
y una discusión de su impacto. Peguen aquí los prompts reales que usaron y
expliquen qué partes del trabajo salieron de la IA, qué cambiaron ustedes y por qué.
-->

**Herramientas usadas:** <herramientas>

**Prompts principales:**

1. `<prompt literal>`
2. `<prompt literal>`

**Impacto en el resultado:** <discusión>

---

## Cacería de la alucinación

Estos son errores que aparecieron mientras desarrollábamos el trabajo con ayuda de IA. Para cada uno damos el prompt, la respuesta, cómo lo sospechamos, la evidencia, la corrección y la lección.

### Hallazgo 1 (datos del mundo real): tarifas de peaje equivocadas en un resumen de búsqueda

1. **Prompt (búsqueda web con resumen automático):** `tarifas peaje 2026 turismo AP-6 AP-66 AP-68 AP-9 AP-71 AP-53 AP-7 importe tramo completo euros`
2. **Respuesta de la IA:** "AP-6 (Villalba-Adanero): 15,70 €; AP-66 (León-Campomanes): 9,20 €; AP-68 (Bilbao-Zaragoza): 39,90 €; AP-9 (Ferrol-Frontera Portuguesa): 21,30 €; **AP-71 (León-Astorga): 4,10 €**; **AP-53 (Santiago-Alto de Santo Domingo): 5,90 €**; AP-7 (Málaga-Guadiaro): 9,50 €".
3. **Cómo lo sospechamos:** el resumen no citaba la tabla de donde sacaba cada número, y sabíamos que la AP-71 tiene tarifa diurna y nocturna, pero la respuesta daba un solo valor.
4. **Evidencia:** descargamos los PDF oficiales del Ministerio de Transportes ("Peajes vigentes desde el día 01/01/2026"). AP-6 (15,70 €) y AP-68 (39,90 €) coinciden. Pero la AP-71 León-Astorga cuesta **6,20 € de día y 2,50 € de noche** para ligeros, y ninguno de los dos es 4,10 €. La AP-53 Santiago-Alto de Santo Domingo cuesta **7,25 €**, no 5,90 €.
5. **Corrección:** en `datos/peajes_tarifas.csv` solo usamos importes leídos directamente de los PDF oficiales. Las autopistas que no pudimos verificar usan el promedio y quedan marcadas.
6. **Lección:** un resumen generado por IA puede mezclar números correctos con incorrectos en la misma lista, y que dos sean correctos no garantiza los demás. Hay que verificar cada número contra la fuente primaria.

### Hallazgo 2 (código): suponer que OSRM marca los tramos de peaje

1. **Prompt:** `<prompt literal con el que se pidió el script de descarga de datos>`
2. **Respuesta de la IA:** la primera versión del script detectaba los peajes buscando la etiqueta `"toll"` en el campo `classes` de las intersecciones de cada paso de la ruta de OSRM.
3. **Cómo lo sospechamos:** probamos con rutas que seguro usan autopistas de pago (León-Oviedo por la AP-66, Bilbao-San Sebastián por la AP-8) y el script decía 0 km de peaje en todas.
4. **Evidencia:** al imprimir los pasos de la ruta León-Oviedo, el paso "AP-66 Autopista Ruta de la Plata" (28,6 km) tenía `classes` vacío. El servidor público de OSRM no devuelve esa clase. Además, la consulta con `exclude=toll` devuelve el error `"Exclude flag combination is not supported."`.
5. **Corrección:** usamos el código de la vía (`ref`) de cada paso y lo cruzamos con la lista de autopistas que siguen siendo de pago en 2026.
6. **Lección:** probar con un caso cuya respuesta se conoce (una ruta que seguro paga peaje) antes de correr las 1081 rutas.

### Hallazgo 3 (código / datos): kilómetros de peaje perdidos por una etiqueta de OpenStreetMap

1. **Prompt:** `<prompt literal>`
2. **Respuesta de la IA:** el script guardaba solo la primera referencia de cada paso (`ref.split(";")[0]`) y contaba como peaje solo los pasos con código exacto de autopista de pago.
3. **Cómo lo sospechamos:** la ruta León-Oviedo daba 28,7 km de peaje, pero la AP-66 entre León y Campomanes tiene unos 78 km.
4. **Evidencia:** en el sentido Oviedo → León, OSRM devuelve el tramo de 47,7 km con `ref = "A-66"` pero con nombre "Autopista Ruta de la Plata"; en el otro sentido el mismo tramo aparece como `AP-66`. Es decir, la etiqueta en OpenStreetMap es inconsistente, y el código no lo tenía en cuenta.
5. **Corrección:** ahora guardamos la referencia completa y el nombre de la vía (`datos/km_por_via.csv`), y la función `via_de_peaje` de `tsp/datos.py` cuenta como AP-xx un tramo "A-xx" cuyo nombre dice "Autopista" cuando la AP-xx es de pago.
6. **Lección:** revisar los datos con casos conocidos, en los dos sentidos del viaje, y no asumir que los datos abiertos están etiquetados de forma consistente.

### Hallazgo 4 (código / conceptos): algoritmo genético que se estancaba por la búsqueda local

1. **Prompt:** `<prompt literal>`
2. **Respuesta de la IA:** la primera versión del GA para el TSP aplicaba 2-opt solo al mejor individuo de cada generación, con la idea de que eso bastaba para que el GA "afinara" la solución.
3. **Cómo lo sospechamos:** en una instancia de prueba (47 puntos aleatorios) el GA se quedaba en el mismo costo desde la generación 0 hasta la 300 (una sola mejora en la historia).
4. **Evidencia:** el GA llegaba a 646,3 y la colonia de hormigas a 574,5 en la misma instancia. El mejor individuo, ya optimizado con 2-opt, era tan bueno que ningún hijo producido por cruce OX + inversión lo superaba, así que el elitismo lo mantenía para siempre.
5. **Corrección:** aplicar 2-opt a cada hijo con probabilidad 0,05 (`--prob-2opt`) y subir el torneo a 5. Con eso el GA llega también a 574,5.
6. **Lección:** mirar la curva de convergencia y no solo el valor final. Un algoritmo que no mejora nunca después de la primera generación tiene un problema de diseño.

### Hallazgo 5

<!-- COMPLETAR POR EL EQUIPO con un hallazgo propio. Se piden al menos cinco y
al menos tres categorías (aquí ya hay datos, código y conceptos).
Ideas para revisar: constantes de PSO, mínimo global y dominio de las funciones
en lo que les responda una IA, referencias bibliográficas (que existan y que el
DOI resuelva), afirmaciones sobre convergencia garantizada de GA/PSO/ACO. -->

### Verificaciones que no encontraron errores

- Gradientes analíticos de Rosenbrock y Rastrigin contra diferencias centrales en 100 puntos para $n=2,3,5$: error relativo máximo $\sim10^{-10}$ (`python pruebas.py`).
- Mínimos globales: $f(1,\dots,1)=0$ en Rosenbrock y $f(\mathbf 0)=0$ en Rastrigin, con gradiente nulo (`pruebas.py`).
- Contadores de evaluaciones: prueba con 11 evaluaciones de $f$ y 1 de $\nabla f$ en 3D, que debe dar 17 equivalentes (`pruebas.py`).
- Peaje Bilbao-Zaragoza calculado por nuestro modelo: 39,79 €, contra 39,90 € de la tabla oficial de la AP-68.
- Precio del gasóleo: un resumen de búsqueda daba 1,834 €/l (citando el Boletín Petrolero del 14/09/2026 a través de la prensa). No pudimos comprobar esa cifra en la fuente, así que usamos el archivo oficial más reciente del Weekly Oil Bulletin: 1 934,43 € por 1000 l (precios al 28/09/2026).

---

## Referencias

Clerc, M., & Kennedy, J. (2002). The particle swarm: Explosion, stability, and convergence in a multidimensional complex space. *IEEE Transactions on Evolutionary Computation, 6*(1), 58–73. https://doi.org/10.1109/4235.985692

Comisión Europea. (2026). *Weekly Oil Bulletin* [Precios de los combustibles con impuestos, 28 de septiembre de 2026]. Recuperado el 6 de octubre de 2026, de https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en

Croes, G. A. (1958). A method for solving traveling-salesman problems. *Operations Research, 6*(6), 791–812. https://doi.org/10.1287/opre.6.6.791

Davis, L. (1985). Applying adaptive algorithms to epistatic domains. En *Proceedings of the 9th International Joint Conference on Artificial Intelligence* (Vol. 1, pp. 162–164). Morgan Kaufmann.

Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: Optimization by a colony of cooperating agents. *IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 26*(1), 29–41. https://doi.org/10.1109/3477.484436

Dorigo, M., & Stützle, T. (2004). *Ant colony optimization*. MIT Press.

Goldberg, D. E. (1989). *Genetic algorithms in search, optimization, and machine learning*. Addison-Wesley.

Griewank, A., & Walther, A. (2008). *Evaluating derivatives: Principles and techniques of algorithmic differentiation* (2.ª ed.). SIAM. https://doi.org/10.1137/1.9780898717761

Holland, J. H. (1975). *Adaptation in natural and artificial systems*. University of Michigan Press.

Jamil, M., & Yang, X.-S. (2013). A literature survey of benchmark functions for global optimisation problems. *International Journal of Mathematical Modelling and Numerical Optimisation, 4*(2), 150–194. https://doi.org/10.1504/IJMMNO.2013.055204

Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. En *Proceedings of ICNN'95 – International Conference on Neural Networks* (Vol. 4, pp. 1942–1948). IEEE. https://doi.org/10.1109/ICNN.1995.488968

km77.com. (2024). *SEAT León 2.0 TDI 85 kW (115 CV) Style (2024): precio y ficha técnica*. Recuperado el 6 de octubre de 2026, de https://www.km77.com/coches/seat/leon/2020/5-puertas/estandar/leon-20-tdi-85-kw-115-cv-style/datos

Luxen, D., & Vetter, C. (2011). Real-time routing with OpenStreetMap data. En *Proceedings of the 19th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems* (pp. 513–516). ACM. https://doi.org/10.1145/2093973.2094062

Ministerio de Transportes y Movilidad Sostenible. (2026). *Peajes vigentes desde el día 01/01/2026* [Tablas de tarifas de las autopistas AP-6, AP-9, AP-53, AP-68 y AP-71]. Recuperado el 6 de octubre de 2026, de https://www.transportes.gob.es/carreteras/nuestra-red/autopistas-de-peaje/peajes-actuales

Natural Earth. (s. f.). *Admin 0 – Countries* (escala 1:50m) [Datos geográficos]. Recuperado el 6 de octubre de 2026, de https://www.naturalearthdata.com

Nocedal, J., & Wright, S. J. (2006). *Numerical optimization* (2.ª ed.). Springer. https://doi.org/10.1007/978-0-387-40065-5

OpenStreetMap contributors. (2026). *Nominatim* [Servicio de geocodificación]. Recuperado el 6 de octubre de 2026, de https://nominatim.openstreetmap.org

Rastrigin, L. A. (1974). *Systems of extremal control*. Nauka.

Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least value of a function. *The Computer Journal, 3*(3), 175–184. https://doi.org/10.1093/comjnl/3.3.175

Storn, R., & Price, K. (1997). Differential evolution – A simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization, 11*(4), 341–359. https://doi.org/10.1023/A:1008202821328
