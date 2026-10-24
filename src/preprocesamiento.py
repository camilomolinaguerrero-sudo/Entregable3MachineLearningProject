"""Preprocesamiento encapsulado en `ColumnTransformer` (se ajusta solo con entrenamiento).

Hay dos variantes, porque los modelos no necesitan lo mismo:

* `lineal`: la del Entregable 1 (log1p, winsorización, splines en las razones de cambio,
  escalado y one-hot) más las variables espaciales nuevas. La usan los modelos sensibles a la
  escala o a la forma funcional: regresión logística, Ridge, Lasso, k-NN, Naive Bayes y SVM/SVR.
* `arboles`: imputación y one-hot, sin transformaciones monótonas ni escalado, que no cambian las
  particiones de un árbol. La usan Decision Tree, Random Forest y XGBoost.

En las dos, el municipio de origen entra con codificación por objetivo (`TargetEncoder`), que
estima la tasa histórica del municipio con ajuste cruzado interno para no filtrar el objetivo de
cada fila en su propia codificación.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, SplineTransformer,
                                   StandardScaler, TargetEncoder)

from config import BIN, CAT, ESP_DIN, ESP_TE, LOG, NUM_E1, SEMILLA, SPL


class Winsorizador(BaseEstimator, TransformerMixin):
    """Recorta cada columna en los cuantiles [inferior, superior] aprendidos en `fit`.

    Parameters
    ----------
    inferior, superior : cuantiles de recorte (por defecto 1 % y 99 %).
    """

    def __init__(self, inferior: float = 0.01, superior: float = 0.99):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lim_inf_ = np.nanquantile(X, self.inferior, axis=0)
        self.lim_sup_ = np.nanquantile(X, self.superior, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lim_inf_, self.lim_sup_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _te(tarea: str):
    tipo = "binary" if tarea == "clf" else "continuous"
    return TargetEncoder(target_type=tipo, smooth="auto", cv=5, shuffle=True, random_state=SEMILLA)


def prep_lineal(tarea: str = "clf", espaciales: bool = True) -> ColumnTransformer:
    """Preprocesador para modelos lineales, de distancias y de núcleo.

    Parameters
    ----------
    tarea : "clf" o "reg" (cambia el tipo de la codificación por objetivo).
    espaciales : si False reproduce exactamente el preprocesador del Entregable 1.
    """
    num = NUM_E1 + (ESP_DIN if espaciales else [])
    pasos = [
        ("log", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("win", Winsorizador()), ("esc", StandardScaler())]), LOG),
        ("spl", Pipeline([("imp", SimpleImputer(strategy="median")), ("win", Winsorizador()),
                          ("spl", SplineTransformer(n_knots=5, degree=3, knots="quantile")),
                          ("esc", StandardScaler())]), SPL),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("win", Winsorizador()),
                          ("esc", StandardScaler())]), num),
        ("bin", "passthrough", BIN),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                              sparse_output=False), CAT),
    ]
    if espaciales:
        pasos.append(("te", Pipeline([("te", _te(tarea)), ("esc", StandardScaler())]), ESP_TE))
    return ColumnTransformer(pasos, sparse_threshold=0.0, verbose_feature_names_out=True)


def prep_arboles(tarea: str = "clf", espaciales: bool = True) -> ColumnTransformer:
    """Preprocesador para árboles: imputación por mediana, one-hot y codificación por objetivo."""
    num = LOG + SPL + NUM_E1 + (ESP_DIN if espaciales else [])
    pasos = [
        ("num", SimpleImputer(strategy="median"), num),
        ("bin", "passthrough", BIN),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                              sparse_output=False), CAT),
    ]
    if espaciales:
        pasos.append(("te", _te(tarea), ESP_TE))
    return ColumnTransformer(pasos, sparse_threshold=0.0, verbose_feature_names_out=True)


def preprocesador(tipo: str, tarea: str = "clf", espaciales: bool = True) -> ColumnTransformer:
    """Devuelve el preprocesador `tipo` ("lineal" o "arboles") sin ajustar."""
    if tipo == "lineal":
        return prep_lineal(tarea, espaciales)
    if tipo == "arboles":
        return prep_arboles(tarea, espaciales)
    raise ValueError(tipo)


def nombres_limpios(ct: ColumnTransformer) -> list[str]:
    """Nombres de salida sin el prefijo del bloque (`num__`, `cat__`, ...)."""
    import re
    return [re.sub(r"^(log|spl|num|bin|cat|te)__", "", n) for n in ct.get_feature_names_out()]
