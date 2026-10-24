# %% [markdown]
# # 9. Reproducibilidad y bibliografía
#
# ## 9.1 Entorno
#
# * Python 3.11.16 en macOS (Apple Silicon, 10 núcleos, 24 GB de RAM). El dashboard corre en Render con
#   Python 3.11.16 y las versiones exactas de `requirements.txt`; el entorno completo de los experimentos
#   y del libro está en `requirements-dev.txt`.
# * Semilla global 42 (`src/config.py`), propagada a numpy, scikit-learn, XGBoost, imbalanced-learn,
#   Optuna y DEAP. El AJ-KL es determinista (no tiene pasos aleatorios); la semilla solo fija el TPE que
#   elige sus hiperparámetros y la submuestra del control de 40 mil filas.
# * Los datos crudos del panel están en `data/raw/` (48 MB); las predicciones por fila de los 140 modelos
#   del Entregable 2 (300 MB) no se versionan en GitHub y se regeneran con los scripts de ese entregable.
#
# ## 9.2 Orden de ejecución
#
# | Paso | Script | Salida | Tiempo aproximado |
# |---|---|---|---|
# | 1 | Entregable 2: `scripts/03_preparar_pliegues.py` … `10_comparacion.py` | pliegues, tabla maestra de 140 modelos y predicciones | 29 h-CPU |
# | 2 | `scripts/11_busqueda_literatura.py` y el registro de Web of Science | `data/revision/` | manual |
# | 3 | `scripts/16_sota_sipsa.py <LightGBM\|CatBoost> <pliegue>` | `data/resultados/sota/` | 5-20 min por pliegue |
# | 4 | `scripts/22_ajkl.py <pliegue>` (k1-k4 y final; modelo original) | `data/resultados/ajkl/` | 1-15 min por pliegue |
# | 5 | `scripts/23_externos.py <conjunto>` y `scripts/23b_externos_combinador.py` | `data/resultados/externos/` | 1 h por conjunto |
# | 6 | `scripts/13_artefactos.py todos`, `13b_aligerar.py` y `13c_artefacto_ajkl.py` | `artifacts/`, `artefactos_resumen.csv` | 40 min |
# | 7 | `scripts/14_validacion_estadistica.py` y `14b_validacion_produccion.py` | `data/resultados/validacion/` | 15 min |
# | 8 | `scripts/17_datos_dashboard.py` (EDA y modelos) | `data/dashboard/` | 15 min |
# | 9 | `scripts/20_tabla_maestra.py` | `data/resultados/tabla_maestra.csv` | segundos |
# | 10 | `python src/py2nb.py cuadernos/X.py book/X.ipynb --ejecutar` y `jupyter-book build book/` | `book/_build/html` | 5 min |
#
# Las pruebas unitarias se corren con `pytest tests/`: verifican las propiedades del AJ-KL (aprendizaje en
# datos sintéticos con solución conocida, reproducibilidad, entradas inválidas, compatibilidad con
# `sklearn.base.clone`, óptimo del descenso espejo frente a SLSQP y L-BFGS, casos límite), el modelo
# exploratorio EJUL, la explicación SHAP por permutaciones y la prueba de ida y vuelta de los artefactos.
# Los scripts `12_modelo_original.py`, `15_ablacion.py`, `18_complejidad.py`, `19_robustez.py` y
# `21_ejul_catboost.py` corresponden al modelo exploratorio EJUL.

# %%
import subprocess
import sys

import pandas as pd
from IPython.display import display

r = subprocess.run([sys.executable, "-m", "pytest", "../tests", "-q", "--no-header", "-p", "no:cacheprovider"],
                   capture_output=True, text=True)
print(r.stdout[-1500:])

# %% [markdown]
# ## 9.3 Tabla maestra de experimentos
#
# La tabla maestra reúne las 140 combinaciones del Entregable 2, el modelo original y los dos comparadores
# del estado del arte, con las métricas de cada pliegue externo y de la prueba 2025, los tiempos y la
# semilla. Está en `data/resultados/tabla_maestra.csv`.

# %%
maestra = pd.read_csv("../data/resultados/tabla_maestra.csv")
print(maestra.shape)
cols = ["tarea", "modelo", "balanceo", "optimizador", "pliegue", "auc", "auc_pr", "brier", "ece", "f1", "rmse", "t_total", "semilla"]
display(maestra[[c for c in cols if c in maestra.columns]].tail(15))

# %% [markdown]
# ## 9.4 Bibliografía
#
# ```{bibliography}
# :all:
# ```
