"""Configuración global del Entregable 2: rutas, semilla, partición y listas de variables.

Todas las rutas se resuelven desde la raíz del proyecto para que los scripts, los procesos
paralelos de joblib y los cuadernos del libro lean exactamente los mismos archivos.
"""
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
RES = ROOT / "data" / "resultados"
FIG = ROOT / "book" / "figuras"
PANEL = RAW / "panel_corredores.parquet"

# Semilla global única (sección 7.3 de la guía): se propaga a numpy, random, scikit-learn,
# XGBoost, imbalanced-learn, Optuna y DEAP a través de esta constante.
SEMILLA = 42


def fijar_semillas(semilla: int = SEMILLA) -> None:
    """Fija las semillas de `random`, `numpy` y `PYTHONHASHSEED` en el proceso actual."""
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)


# Partición cronológica heredada del Entregable 1: entrenamiento hasta la semana objetivo del
# 27-10-2024 y prueba con todas las semanas objetivo de 2025 (hueco de 10 semanas).
FIN_TRAIN = pd.Timestamp("2024-10-27")
INICIO_TEST = pd.Timestamp("2025-01-06")

# Validación anidada (sección 3.2 de la guía)
N_EXTERNOS = 4            # pliegues externos: los cuatro últimos años del periodo de entrenamiento
SEMANAS_EXTERNO = 52      # cada pliegue externo valida sobre 52 semanas completas
N_INTERNOS = 3            # pliegues internos de ventana creciente dentro de cada entrenamiento externo
HUECO = 8                 # semanas de separación (longitud de la ventana de referencia base8)
N_MUESTRA = 40_000        # filas de entrenamiento por pliegue externo para la búsqueda combinatoria
PRESUPUESTO = 24          # evaluaciones por búsqueda (igual para los cuatro optimizadores)

UMBRAL_INTERRUPCION = 0.5

OBJ_CLF = "interrupcion_t1"
OBJ_REG = "y_reg"         # log(1 + razón de abastecimiento en t+1 frente a base8)

# Variables del Entregable 1
LOG = ["n_envios8", "n_alimentos8", "dist_km", "antig_sem"]
SPL = ["ratio_t", "ratio_t_1"]
NUM_E1 = ["log_base8", "cv8", "pend8_rel", "activas8", "activas26", "frac_caida_mercado_t",
          "frac_caida_nacional_t", "festivos_t1", "sem_sin", "sem_cos", "sh_verduras",
          "sh_frutas", "sh_tuberculos", "sh_granos", "sh_lacteos", "sh_carnes", "sh_pescados"]
BIN = ["mismo_depto", "local", "pdet_origen"]
CAT = ["mercado", "depto_origen", "grupo_dom"]

# Variables espaciales nuevas (respuesta a la retroalimentación del Entregable 1). Todas usan
# solo información disponible al cierre de la semana t.
ESP_DIN = ["frac_caida_mpio_t", "frac_caida_vec_t", "tasa_int_vec26", "tasa_int_mpio26"]
ESP_TE = ["cod_mpio"]     # codificación por objetivo del municipio de origen (dentro del pipeline)

NUM = NUM_E1 + ESP_DIN
X_COLS = LOG + SPL + NUM + BIN + CAT + ESP_TE
