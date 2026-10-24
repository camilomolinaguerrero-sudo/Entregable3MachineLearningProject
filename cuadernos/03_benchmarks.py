# %% [markdown]
# # 3. Modelos benchmark: selección, reentrenamiento y artefactos
#
# Del universo de 140 combinaciones del Entregable 2, el tablero no carga todas: carga la mejor
# combinación de cada uno de los siete modelos de clasificación, elegida con un criterio estadístico
# (guía, sección 4). Este capítulo documenta la selección, el reentrenamiento con todo el periodo de
# entrenamiento, la serialización con sus metadatos y la prueba de ida y vuelta, y añade dos
# comparadores recientes del estado del arte (LightGBM y CatBoost) evaluados con el mismo protocolo.

# %%
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)
RES = Path("../data/resultados")
ART = Path("../artifacts")

# %% [markdown]
# ## 3.1 Criterio de selección
#
# Para cada modelo se elige la combinación (balanceo × optimizador) con mayor AUC medio en los cuatro
# pliegues externos de la validación anidada, que es la métrica de selección del Entregable 2. Los
# empates exactos se resuelven con el Brier medio (capítulo 1). Para la mejor combinación global se
# verifica que pertenezca al grupo de finalistas de Friedman-Nemenyi del Entregable 2: su rango medio
# entre las 112 combinaciones de clasificación, en 48 bloques mensuales, debe estar a menos de una
# diferencia crítica del mejor rango.

# %%
sel = pd.read_csv(RES / "seleccion_mejores.csv")
tabla = pd.DataFrame({
    "Modelo": sel["modelo"], "Balanceo": sel["balanceo"], "Optimizador": sel["optimizador"],
    "AUC externo": [f"{m:.4f} ± {d:.4f}" for m, d in zip(sel["auc_media"], sel["auc_de"])],
    "AUC-PR": [f"{m:.4f} ± {d:.4f}" for m, d in zip(sel["auc_pr_media"], sel["auc_pr_de"])],
    "Brier": [f"{m:.4f} ± {d:.4f}" for m, d in zip(sel["brier_media"], sel["brier_de"])],
    "ECE": [f"{m:.4f} ± {d:.4f}" for m, d in zip(sel["ece_media"], sel["ece_de"])],
    "F1": [f"{m:.4f} ± {d:.4f}" for m, d in zip(sel["f1_media"], sel["f1_de"])],
    "Rango Friedman (112)": sel["rango_friedman"].round(2), "Finalista Nemenyi": sel["finalista_nemenyi"],
    "Motivo": sel["motivo"]})
display(tabla)
cd = float(pd.read_parquet(RES / "comp_friedman_global.parquet")["CD"].iloc[0])
display(Markdown(f"La diferencia crítica de Nemenyi entre 112 combinaciones y 48 bloques es {cd:.2f} rangos. "
                 f"XGBoost sin balanceo con Random Search tiene el mejor rango ({sel['rango_friedman'].iloc[0]:.2f}) y, "
                 "por definición, pertenece al grupo de finalistas; Random Forest y la SVM también quedan a menos de una "
                 "diferencia crítica, mientras que la regresión logística, k-NN, Decision Tree y Naive Bayes quedan fuera."))

# %% [markdown]
# Las ventajas entre combinaciones de un mismo modelo son pequeñas (columna «Motivo»): la superficie del
# AUC es plana cerca del óptimo, como mostró el Entregable 2, y por eso la elección del balanceo pesa más
# que la del optimizador. Solo Decision Tree y Naive Bayes se benefician de una técnica de balanceo.
#
# ## 3.2 Reentrenamiento y serialización
#
# Cada tubería completa (preprocesamiento, balanceo cuando corresponde y modelo) se reentrena con las
# 622.244 corredor-semanas del periodo de entrenamiento, con los hiperparámetros que el bucle interno
# eligió en el pliegue final, y se serializa con `joblib.dump(..., compress=3)` en `artifacts/<modelo>/`.
# Dos decisiones:
#
# * **k-NN** guarda todo su conjunto de referencia. Con 622 mil filas y unas 70 variables el artefacto
#   ocuparía unos 350 MB, más que la memoria del plan gratuito de Render, y su número de vecinos
#   (k = 301) se eligió para la densidad de una muestra de 40 mil filas. Se usa como referencia la muestra
#   del pliegue final, la misma con que se ajustaron sus hiperparámetros.
# * **La SVM** (Nyström + LinearSVC) no produce probabilidades. Se envuelve en una clase propia
#   (`ConPlatt`) que ajusta el escalado de Platt con puntajes fuera de pliegue, y su umbral de decisión se
#   lleva a la escala de probabilidad con esa misma transformación monótona.
#
# Todos los artefactos se guardan con un solo hilo de cómputo para que sus predicciones sean
# deterministas.

# %%
art = pd.read_csv(RES / "artefactos_resumen.csv")
display(art[["modelo", "filas", "segundos_fit", "ms_por_1000", "mb", "ida_y_vuelta"]].rename(columns={
    "modelo": "Modelo", "filas": "Filas de entrenamiento", "segundos_fit": "Entrenamiento (s)",
    "ms_por_1000": "Inferencia (ms por 1000 filas)", "mb": "Tamaño (MB)", "ida_y_vuelta": "Ida y vuelta idéntica"}).round(3))

# %% [markdown]
# ## 3.3 Metadatos y prueba de ida y vuelta
#
# Junto a cada artefacto hay un `metadata.json` con las versiones de las librerías, la semilla, los
# hiperparámetros finales, el esquema de las 34 columnas de entrada (nombre, tipo y rango válido o
# categorías) y las métricas del bucle externo y de la prueba 2025. El tablero construye su formulario de
# predicción a partir de ese esquema. La prueba de ida y vuelta predice una muestra fija de 2.000
# corredor-semanas de 2025 con el modelo en memoria, lo serializa, lo vuelve a cargar y exige
# predicciones idénticas (`np.array_equal`); `tests/test_artefactos.py` repite la comprobación contra las
# predicciones guardadas cada vez que se corre la batería de pruebas.

# %%
meta = json.loads((ART / "xgboost" / "metadata.json").read_text())
print(json.dumps({k: meta[k] for k in ["modelo", "combinacion", "semilla", "librerias", "hiperparametros",
                                        "filas_entrenamiento", "prueba_ida_y_vuelta"]}, indent=2, ensure_ascii=False))
print("Esquema (primeras 3 columnas):", json.dumps(meta["esquema_entrada"][:3], ensure_ascii=False)[:600], "...")

# %% [markdown]
# ## 3.4 Desempeño en 2025 con todo el entrenamiento

# %%
cols = ["auc", "auc_pr", "brier", "log_loss", "ece", "f1", "precision", "recall", "accuracy"]
t25 = art.set_index("modelo")[[f"{c}_2025" for c in cols]]
t25.columns = cols
display(t25.sort_values("auc", ascending=False).round(4))

# %% [markdown]
# ## 3.5 Comparadores recientes del estado del arte
#
# La guía pide comparar el modelo original también con métodos recientes que no se vieron en clase.
# LightGBM {cite}`ke2017lightgbm` y CatBoost {cite}`prokhorenkova2018catboost` se evaluaron con
# exactamente el mismo protocolo que los 140 modelos y que el modelo original: los mismos pliegues
# externos e internos, la misma muestra de 40 mil filas, el mismo preprocesamiento, TPE con 24
# evaluaciones y semilla 42, y el AUC interno como criterio (`scripts/16_sota_sipsa.py`).

# %%
try:
    import joblib
    filas = []
    for m in ["LightGBM", "CatBoost"]:
        for pl in ["k1", "k2", "k3", "k4", "final"]:
            r = joblib.load(RES / "sota" / f"{m}_{pl}.joblib")["resumen"]
            filas.append({"modelo": m, "pliegue": pl, **{c: r[c] for c in cols}, "t_total_s": r["t_total"]})
    sota = pd.DataFrame(filas)
    ext = sota[sota.pliegue != "final"].groupby("modelo")[cols].agg(["mean", "std"]).round(4)
    display(ext)
    display(sota[sota.pliegue == "final"].set_index("modelo")[cols].round(4))
except FileNotFoundError:
    print("Resultados de LightGBM y CatBoost pendientes.")
