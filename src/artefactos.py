"""Tuberías completas (preprocesamiento + balanceo + modelo) para serializar y cargar en el dashboard.

Cada artefacto recibe un DataFrame con las columnas crudas de `config.X_COLS` y devuelve la
probabilidad de interrupción en la semana siguiente. Ningún artefacto se reentrena en el servidor.
"""
from __future__ import annotations

import json
import platform
import sys
from importlib import metadata as md

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold

import config as C
import evaluacion as EV
import modelos as M
import preprocesamiento as P

COLS_GRUPO = ["mercado", "cod_mpio", "depto_origen"]
LIBRERIAS = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "imbalanced-learn", "joblib"]


class CodigosGrupo(BaseEstimator, TransformerMixin):
    """Códigos enteros de corredor (central|municipio), municipio y departamento de origen.

    Solo usa las etiquetas (no el objetivo). Las categorías no vistas en `fit` reciben −1, que el
    codificador jerárquico del EJUL interpreta como "sin grupo" y reemplaza por la tasa del nivel
    superior.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=COLS_GRUPO).astype(str)
        claves = self._claves(X)
        self.mapas_ = [{v: i for i, v in enumerate(pd.unique(c))} for c in claves]
        return self

    @staticmethod
    def _claves(X):
        return [X["mercado"] + "|" + X["cod_mpio"], X["cod_mpio"], X["depto_origen"]]

    def transform(self, X):
        X = pd.DataFrame(X, columns=COLS_GRUPO).astype(str)
        return np.column_stack([c.map(m).fillna(-1).to_numpy(np.int64) for c, m in zip(self._claves(X), self.mapas_)])

    def get_feature_names_out(self, input_features=None):
        return np.array(["g_corredor", "g_mpio", "g_depto"], dtype=object)


def prep_ejul() -> ColumnTransformer:
    """Preprocesador de árboles con las tres columnas de grupo al final."""
    base = P.preprocesador("arboles", "clf")
    return ColumnTransformer(list(base.transformers) + [("grupos", CodigosGrupo(), COLS_GRUPO)],
                             sparse_threshold=0.0, verbose_feature_names_out=True)


class ConPlatt(BaseEstimator, ClassifierMixin):
    """Añade probabilidades a un clasificador de margen (LinearSVC) con escalado de Platt.

    El escalado se ajusta con puntajes fuera de pliegue (validación cruzada estratificada) para no
    reutilizar los puntajes de entrenamiento, que son optimistas.
    """

    def __init__(self, estimador=None, pliegues=3, random_state=C.SEMILLA):
        self.estimador = estimador
        self.pliegues = pliegues
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        oof = np.empty(len(y))
        for tr, va in StratifiedKFold(self.pliegues, shuffle=True, random_state=self.random_state).split(X, y):
            oof[va] = clone(self.estimador).fit(X[tr], y[tr]).decision_function(X[va])
        self.platt_ = EV.Platt().fit(oof, y)
        self.estimador_ = clone(self.estimador).fit(X, y)
        return self

    def decision_function(self, X):
        return self.estimador_.decision_function(X)

    def predict_proba(self, X):
        p = self.platt_.predict(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def tuberia(modelo: str, balanceo: str, params: dict, y=None, n_jobs: int = 8):
    """Tubería sin ajustar de un modelo benchmark del Entregable 2."""
    from imblearn.pipeline import Pipeline as PipelineImb
    espec = M.catalogo("clf")[modelo]
    est = M.construir(espec, params, balanceo, y)
    if "n_jobs" in est.get_params() and modelo != "KNN":
        est.set_params(n_jobs=n_jobs)
    if modelo == "SVM":
        est = ConPlatt(est)
    pasos = [("prep", P.preprocesador(espec.prep, "clf"))]
    if balanceo in ("SMOTE", "ADASYN"):
        from imblearn.over_sampling import ADASYN, SMOTE
        pasos.append(("balanceo", (SMOTE if balanceo == "SMOTE" else ADASYN)(random_state=C.SEMILLA)))
    pasos.append(("modelo", est))
    return PipelineImb(pasos)


def esquema_entrada(df: pd.DataFrame) -> list[dict]:
    """Nombre, tipo y rango válido (mínimo y máximo de entrenamiento o categorías) de cada columna."""
    salida = []
    for c in C.X_COLS:
        s = df[c]
        if c in C.CAT + C.ESP_TE:
            cats = s.astype(str).value_counts()
            salida.append({"nombre": c, "tipo": "categórica", "categorias": cats.index.tolist(),
                           "frecuencias": cats.astype(int).tolist()})
        elif c in C.BIN:
            salida.append({"nombre": c, "tipo": "binaria", "valores": [0, 1]})
        else:
            v = pd.to_numeric(s, errors="coerce")
            salida.append({"nombre": c, "tipo": "numérica", "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
                           "p01": float(np.nanquantile(v, 0.01)), "p99": float(np.nanquantile(v, 0.99)),
                           "mediana": float(np.nanmedian(v)), "faltantes": float(v.isna().mean())})
    return salida


def versiones() -> dict:
    out = {"python": platform.python_version(), "plataforma": platform.platform()}
    for lib in LIBRERIAS:
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            out[lib] = None
    return out


def guardar_json(obj, ruta):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    ruta.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv))


def raiz_sys_path():
    """Ruta de `src` para que los artefactos encuentren sus clases al deserializar."""
    return str(C.ROOT / "src") if str(C.ROOT / "src") not in sys.path else None


class AJKLProduccion(BaseEstimator, ClassifierMixin):
    """Tubería de producción del AJ-KL: modelos base ya entrenados + combinador jerárquico.

    Recibe el DataFrame crudo con las columnas de `config.X_COLS`, calcula la probabilidad de cada modelo
    base, los códigos de central, central × grupo y corredor (con los mismos diccionarios del
    entrenamiento; las etiquetas no vistas reciben −1 y heredan los pesos del nodo padre) y aplica el
    combinador AJ-KL.
    """

    def __init__(self, bases=None, combinador=None, mapas=None, orden=None):
        self.bases = bases
        self.combinador = combinador
        self.mapas = mapas
        self.orden = orden

    def fit(self, X=None, y=None):
        self.classes_ = np.array([0, 1])
        return self

    @staticmethod
    def claves(X):
        X = pd.DataFrame(X)
        c = X["mercado"].astype(str)
        cg = c + "|" + X["grupo_dom"].astype(str)
        return [c, cg, cg + "|" + X["cod_mpio"].astype(str)]

    def probabilidades_base(self, X) -> np.ndarray:
        return np.column_stack([self.bases[m].predict_proba(X)[:, 1] for m in self.orden])

    def codigos(self, X) -> np.ndarray:
        return np.column_stack([k.map(m).fillna(-1).to_numpy(np.int64) for k, m in zip(self.claves(X), self.mapas)])

    def matriz(self, X) -> np.ndarray:
        return np.c_[np.clip(self.probabilidades_base(X), 1e-4, 1 - 1e-4), self.codigos(X)]

    def predict_proba(self, X):
        return self.combinador.predict_proba(self.matriz(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def pesos(self, X) -> np.ndarray:
        """Pesos de los modelos base en el nodo que usa cada fila (interpretación local)."""
        return self.combinador.pesos_por_fila(self.matriz(X))
