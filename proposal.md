# Proposal — Proyecto final de curso: Machine Learning

## 1. Título del proyecto

**Predicción de retrasos de llegada en vuelos que despegan de Nueva York (2013)**

## 2. Integrantes

- Rodrigo de Santa Maria Anco Ito
- Carlos Nayhan Cubas Alcántara
- Eliseo David Velasquez Diaz
- Alejandro Vargas Rios

## 3. Dataset elegido

Se trabaja con **`nycflights13`**, un dataset real y ampliamente documentado
que contiene los **336,776 vuelos** que salieron de los tres aeropuertos de
Nueva York (JFK, LGA, EWR) durante todo 2013, junto con cuatro tablas
relacionadas:

| Tabla | Filas | Contenido |
|---|---|---|
| `flights` | 336,776 | Vuelo individual: fechas, horarios, aerolínea, origen/destino, retrasos |
| `weather` | 26,115 | Clima horario en cada uno de los 3 aeropuertos de origen |
| `planes` | 3,322 | Metadatos de la aeronave (fabricante, modelo, año, asientos) |
| `airports` | 1,458 | Metadatos de aeropuertos (coordenadas, altitud) |
| `airlines` | 16 | Nombre completo de cada aerolínea |

Tras unir las tablas relevantes (`flights` + `planes` + `weather` +
`airlines` + `airports`), el dataset de trabajo tiene **327,346 filas y 39
columnas**.

**Fuente y verificabilidad:** el dataset se origina en datos de la *U.S.
Bureau of Transportation Statistics* y del *NOAA* (clima), empaquetados por
Hadley Wickham como el paquete `nycflights13` (R y Python), ampliamente usado
en cursos de ciencia de datos. Se instala vía `pip install nycflights13`, por
lo que la carga es 100% reproducible sin descargas manuales ni credenciales.

**Por qué cumple el nivel de complejidad pedido (Opción E — dataset
propuesto):**
- ✅ Más de 50,000 filas (327,346) y **múltiples tablas relacionales** (5).
- ✅ Datos **temporales** (todo un año, con estacionalidad y estructura horaria).
- ✅ Fuente verificable (BTS / NOAA, vía paquete documentado).
- ✅ Tarea predictiva clara (retraso de vuelo — problema real de la industria).
- ✅ Dificultad real de limpieza y modelado: valores faltantes no aleatorios,
  necesidad de *joins* entre tablas, alto riesgo de *leakage* temporal (ver
  sección 8), variables categóricas de alta cardinalidad (`dest`, `tailnum`),
  y desbalance de clases.
- ✅ No es un dataset clásico "de juguete" (no es Iris/Titanic/Wine/MNIST).

## 4. Pregunta predictiva

> Dado un vuelo, **antes de que despegue**, ¿es posible determinar si llegará a su destino con un
> retraso de **15 minutos o más** respecto a su horario programado de
> llegada?

Es un problema de **clasificación binaria** (también podría plantearse como
regresión sobre `arr_delay`, pero se elige la versión binaria porque es la
definición estándar de "vuelo retrasado" usada por el Bureau of Transportation
Statistics de EE. UU., y porque produce una tarea con desbalance de clases
real).

## 5. Variable objetivo

`retraso_15` = 1 si `arr_delay >= 15` minutos, 0 en caso contrario.

- **Distribución real:** 75.5% de los vuelos a tiempo vs. 24.5%
  retrasados → desbalance moderado, suficiente para exigir métricas más
  allá de accuracy.
- Los **vuelos cancelados** (2.8% del total, `arr_delay` = NaN) se excluyen
  del target y se documentan como limitación (ver sección 12).

## 6. Unidad de predicción

**Un vuelo individual** (una fila de la tabla `flights`, identificada por
`carrier` + `flight` + `origin` + `time_hour`), en el momento en que se
conoce su horario programado (antes del despegue real).

## 7. Variables disponibles antes de la predicción

Al momento de predicción (antes del despegue) están disponibles, entre otras:

- **Temporales:** `month`, `day`, `hour` (programada), `sched_dep_time`,
  `sched_arr_time`.
- **Vuelo:** `carrier`, `origin`, `dest`, `distance`.
- **Aeronave:** `plane_year`, `manufacturer`, `model`, `engines`, `seats`,
  `engine` (unidas desde `planes` por `tailnum`).
- **Clima en el aeropuerto de origen** a la hora programada de salida:
  `temp`, `dewp`, `humid`, `wind_speed`, `precip`, `pressure`, `visib`
  (unidas desde `weather`).

**Limitación importante:** en este proyecto se
usa el clima *observado* como proxy, aunque en producción real solo se
tendría un *pronóstico* (con su propio error). Es decir, en este modelo se conoce el clima con mayor precisión de lo que se conoce en una situación real, que se basa en la predicción del clima. Por lo tanto, es un error que deberá ser añadido durante pruebas de campo del modelo.

## 8. Riesgos de leakage

Se identificaron y **excluyeron** las columnas que solo se
conocen durante o después del vuelo:

| Columna | Riesgo |
|---|---|
| `dep_time` | Hora real de salida — desconocida antes del despegue |
| `dep_delay` | Retraso real de salida — predictor casi perfecto del retraso de llegada, pero inexistente al momento de predecir |
| `arr_time` | Hora real de llegada — equivalente al target |
| `air_time` | Duración real del vuelo — solo se conoce al aterrizar |
| `arr_delay` | Es la variable con la que se construye el target |

Un riesgo adicional de leakage temporal es de **partición**: como los datos
tienen estructura tempor+al (estacionalidad, tormentas de días específicos),
un split aleatorio filtraría información del futuro al pasado. Por eso se
usa un split temporal (ver sección 10) en lugar de un `train_test_split`
aleatorio.

## 9. Métrica principal y métrica secundaria

- **Métrica principal: F1-score** de la clase "retrasado" — balancea
  precision y recall, apropiado para el desbalance de clases observado (un
  modelo que ignora la clase minoritaria puede tener accuracy alto pero F1
  cercano a 0, como se demuestra empíricamente en el notebook).
- **Métrica secundaria: ROC-AUC** — mide la capacidad de discriminación del
  modelo independientemente del umbral de decisión, útil porque el umbral
  óptimo (15 min) podría ajustarse según el caso de uso.
- Se reporta también **accuracy, precision y recall** por transparencia, pero
  explícitamente **no se usa accuracy como criterio principal**, dado el
  desbalance (75.5%/24.5%).

## 10. Plan de validación

**Split temporal** (no aleatorio), para simular el uso real del modelo
(entrenar con el pasado, predecir el futuro) y evitar leakage temporal:

- **Entrenamiento:** enero–octubre 2013 (273,355 vuelos)
- **Validación:** noviembre 2013 (26,971 vuelos)
- **Test (evaluación final, un solo uso):** diciembre 2013 (27,020 vuelos)

## 11. Modelo baseline

**Regresión logística** con `class_weight='balanced'`, sobre variables
numéricas estandarizadas y variables categóricas codificadas con one-hot
(pipeline con `ColumnTransformer`). Se compara contra un baseline "tonto"
(`DummyClassifier`, siempre predice la clase mayoritaria) para dejar en
evidencia el problema de reportar solo accuracy.

**Resultado en validación (noviembre):**

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy (clase mayoritaria) | 0.819 | 0.000 | 0.000 | **0.000** | 0.500 |
| Regresión logística | 0.671 | 0.291 | 0.568 | **0.385** | 0.673 |

**Resultado en test (diciembre, lectura final):**

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy (clase mayoritaria) | 0.665 | 0.000 | 0.000 | **0.000** | 0.500 |
| Regresión logística | 0.644 | 0.477 | 0.659 | **0.553** | 0.702 |

El baseline honesto tiene *menor* accuracy que el dummy, pero es el único que
realmente detecta vuelos retrasados (F1 y recall muy por encima de 0),
confirmando por qué la consigna penaliza fuertemente reportar solo accuracy
en un problema desbalanceado.

## 12. Riesgos técnicos

- **Clima observado vs. pronóstico:** usar clima real (no pronosticado) hace
  que el desempeño reportado sea optimista respecto a un sistema en
  producción.
- **Vuelos cancelados excluidos del target:** introduce un sesgo de
  selección; un sistema real debería tratar la cancelación como un evento
  aparte (posible trabajo futuro: modelo de 3 clases — a tiempo / retrasado /
  cancelado).
- **Matching imperfecto con `planes`:** ~16% de los vuelos no tienen
  aeronave identificada (`tailnum` sin match), lo que se imputa pero podría
  sesgar el modelo si esos aviones no son un subconjunto aleatorio.
- **Alta cardinalidad de `dest` y `tailnum`:** puede generar features
  dispersos (one-hot con muchas columnas); se evaluará *target encoding* u
  otras alternativas en semanas siguientes.
- **Posible fuga sutil por aerolínea-ruta:** `carrier` puede correlacionar
  con rutas específicas que ya capturan `origin`/`dest`, generando
  colinealidad más que información nueva; se revisará con importancia de
  variables.

## 13. Plan de trabajo semanas restantes

| Semana(s) | Actividad |
|---|---|
| 2–3 | Limpieza y feature engineering (interacciones clima×aeropuerto, franjas horarias, antigüedad de flota) |
| 4–6 | Entrenar y comparar ≥3 familias de modelos (regresión logística, Random Forest, Gradient Boosting) con el mismo split temporal |
| 7–8 | Búsqueda de hiperparámetros (validación en noviembre) |
| 9–10 | Análisis de errores por segmento (aeropuerto, aerolínea, mes) e interpretabilidad (importancias / SHAP) |
| 11–12 | Evaluación final única en test (diciembre), redacción de resultados |
| 13–14 | Discusión de riesgos éticos, sesgos y limitaciones; trabajo futuro |
| 15–16 | Informe final, limpieza de repositorio, presentación |
