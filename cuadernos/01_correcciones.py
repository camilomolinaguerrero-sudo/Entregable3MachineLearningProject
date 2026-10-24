# %% [markdown]
# # 1. Trazabilidad de correcciones
#
# La tabla reúne cada observación con la acción realizada, su ubicación en este libro y la evidencia.
# Incluye tres grupos: las observaciones del profesor al Entregable 1 (atendidas en el Entregable 2 y
# retomadas aquí porque el modelo original aprovecha la dependencia espacial), las observaciones del
# profesor al Entregable 2 y los problemas que el propio equipo encontró en el Entregable 2 al
# preparar este entregable. Una corrección que cambia un resultado obliga a recalcular todo lo que
# depende de él; la última columna dice qué se recalculó.

# %%
import sys
import warnings

import pandas as pd
from IPython.display import Markdown, display

sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 400)

# %%
trazabilidad = pd.read_csv("../data/resultados/trazabilidad.csv")
display(trazabilidad)

# %% [markdown]
# ## 1.1 Cambios en los resultados
#
# * **Mejor combinación de k-NN.** En el Entregable 2, k-NN con votos ponderados por clase y k-NN sin
#   balanceo empataron exactamente en AUC externo (0,7751), porque la ponderación es una
#   transformación monótona de la fracción de vecinos positivos y no cambia el orden de los puntajes.
#   El desempate quedaba al orden alfabético. Ahora se desempata con el Brier medio, que sí distingue
#   la calibración: gana k-NN sin balanceo (Brier 0,1661 frente a 0,1869). El ranking de los siete
#   modelos no cambia, pero el diagrama de diferencia crítica, las pruebas de hipótesis y el tablero
#   usan esta combinación.
# * **Universo de la comparación estadística.** Los 140 modelos del Entregable 2 se comparaban entre
#   sí; ahora el universo de las pruebas del capítulo 6 son las siete mejores combinaciones más el
#   modelo original, sobre los mismos 48 meses del bucle externo y la misma prueba 2025.
# * **Métricas de 2025 con todo el entrenamiento.** El Entregable 2 reentrenó con las 622 mil filas
#   cinco de los siete modelos; aquí se reentrenan los siete (k-NN con su muestra de 40 mil filas, por
#   la razón que explica el capítulo 3) y se reportan sus métricas completas.
