"""Carga y caché de los datos y artefactos que usa el dashboard.

Todo se lee una sola vez por proceso (functools.lru_cache). El servidor nunca entrena modelos ni
optimiza hiperparámetros: solo deserializa los artefactos de artifacts/ y lee data/dashboard/.
Las rutas se pueden redirigir con las variables de entorno DASH_DATA y DASH_ARTIFACTS.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DASH_DATA", RAIZ / "data" / "dashboard"))
ART = Path(os.environ.get("DASH_ARTIFACTS", RAIZ / "artifacts"))
if str(RAIZ / "src") not in sys.path:      # los artefactos referencian clases de src/ al deserializarse
    sys.path.insert(0, str(RAIZ / "src"))

# Orden de presentación: el modelo original primero y luego los benchmarks por AUC externo
ORDEN = ["ajkl_original", "xgboost", "random_forest", "svm", "regresion_logistica", "knn", "decision_tree", "naive_bayes",
         "lightgbm", "catboost"]
NOMBRES = {"ajkl_original": "AJ-KL (original)", "xgboost": "XGBoost", "random_forest": "Random Forest", "svm": "SVM",
           "regresion_logistica": "Regresión logística", "knn": "KNN", "decision_tree": "Decision Tree",
           "naive_bayes": "Naive Bayes", "lightgbm": "LightGBM (estado del arte)", "catboost": "CatBoost (estado del arte)"}
LIBRO = os.environ.get("URL_LIBRO", "https://camilomolinaguerrero-sudo.github.io/Entregable3MachineLearningProject/")
REPO = "https://github.com/camilomolinaguerrero-sudo/Entregable3MachineLearningProject"


def _parquet(nombre: str) -> pd.DataFrame:
    ruta = DATA / f"{nombre}.parquet"
    return pd.read_parquet(ruta) if ruta.exists() else pd.DataFrame()


def _json(nombre: str) -> dict:
    ruta = DATA / f"{nombre}.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


@lru_cache(maxsize=None)
def resumen_eda() -> dict:
    return _json("resumen_eda")


@lru_cache(maxsize=None)
def muestra() -> pd.DataFrame:
    m = _parquet("eda_muestra")
    if len(m):
        m["semana_t1"] = pd.to_datetime(m["semana_t1"])
    return m


@lru_cache(maxsize=None)
def semanal() -> pd.DataFrame:
    s = _parquet("eda_semanal")
    if len(s):
        s["semana_t1"] = pd.to_datetime(s["semana_t1"])
    return s


@lru_cache(maxsize=None)
def faltantes() -> pd.DataFrame:
    return _parquet("eda_faltantes")


@lru_cache(maxsize=None)
def atipicos() -> pd.DataFrame:
    return _parquet("eda_atipicos")


@lru_cache(maxsize=None)
def diccionario() -> pd.DataFrame:
    ruta = DATA / "diccionario.csv"
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


@lru_cache(maxsize=None)
def toneladas() -> pd.DataFrame:
    return _parquet("toneladas_mercado_anio")


@lru_cache(maxsize=None)
def tabla(nombre: str) -> pd.DataFrame:
    """Tablas de resultados de modelos (líderes, predicciones 2025, SHAP, pruebas, ablación...)."""
    return _parquet(nombre)


@lru_cache(maxsize=None)
def resultados() -> dict:
    """Cifras de la sección de modelos (veredicto, CD, pesos del EJUL...)."""
    return _json("resumen_modelos")


def modelos_disponibles() -> list[str]:
    return [m for m in ORDEN if (ART / m / "modelo.joblib").exists()]


@lru_cache(maxsize=None)
def metadata(slug: str) -> dict:
    return json.loads((ART / slug / "metadata.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def modelo(slug: str):
    import joblib
    return joblib.load(ART / slug / "modelo.joblib")


@lru_cache(maxsize=None)
def ejemplos() -> pd.DataFrame:
    """Corredor-semanas reales de 2025 (muestra fija de la prueba de ida y vuelta) para el formulario."""
    ruta = ART / "muestra_referencia.parquet"
    return pd.read_parquet(ruta) if ruta.exists() else pd.DataFrame()


SLUG_COMBINACION = {
    "XGBoost | Sin balanceo | Random Search": "xgboost", "Random Forest | Sin balanceo | Random Search": "random_forest",
    "SVM | Sin balanceo | Bayesiana (TPE)": "svm", "Regresión logística | Sin balanceo | Grid Search": "regresion_logistica",
    "KNN | Sin balanceo | Random Search": "knn", "Decision Tree | class_weight | Grid Search": "decision_tree",
    "Naive Bayes | SMOTE | Grid Search": "naive_bayes", "LightGBM": "lightgbm", "CatBoost": "catboost",
}
