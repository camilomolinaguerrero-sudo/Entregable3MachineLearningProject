"""Catálogo de modelos (Etapa 1), técnicas de balanceo (Etapa 2) y espacios de búsqueda.

Cada modelo se describe con:

* `prep`: el preprocesador que necesita ("lineal" o "arboles", ver `preprocesamiento.py`);
* `espacio`: lista de hiperparámetros con su tipo, rango y escala (logarítmica para C, alpha,
  gamma, learning rate, var_smoothing y tamaños mínimos de hoja) y los niveles de la rejilla;
* `construir(params, balanceo, y)`: devuelve el estimador listo para `fit`.

Los cuatro optimizadores trabajan sobre el MISMO espacio. La rejilla es un diseño factorial de
24 puntos dentro de los mismos rangos (con dos o tres niveles por dimensión, o un solo nivel en las
dimensiones que la rejilla fija), de modo que todos los métodos gastan el mismo presupuesto.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from config import SEMILLA

N_ARBOLES_RF = 100
N_ARBOLES_XGB = 300
N_COMPONENTES_NYSTROEM = 300

BALANCEOS = ["Sin balanceo", "SMOTE", "ADASYN", "class_weight"]
OPTIMIZADORES = ["Grid Search", "Random Search", "Bayesiana (TPE)", "Genética (DEAP)"]


# ----------------------------------------------------------------------------------------------
# Espacio de búsqueda
# ----------------------------------------------------------------------------------------------
@dataclass
class Param:
    """Un hiperparámetro del espacio de búsqueda.

    Attributes
    ----------
    nombre : nombre del argumento del estimador.
    tipo : "float", "int" o "cat".
    bajo, alto : límites (para float e int).
    log : si el rango se recorre en escala logarítmica.
    opciones : valores posibles (para cat).
    rejilla : niveles que usa Grid Search.
    """
    nombre: str
    tipo: str
    bajo: float | None = None
    alto: float | None = None
    log: bool = False
    opciones: tuple = ()
    rejilla: tuple = ()

    def decodificar(self, g: float):
        """Convierte un gen en [0, 1] en un valor del hiperparámetro (codificación de la AG y
        del muestreo aleatorio: uniforme en [0, 1] equivale a uniforme o log-uniforme en el rango)."""
        g = float(np.clip(g, 0.0, 1.0))
        if self.tipo == "cat":
            return self.opciones[min(int(g * len(self.opciones)), len(self.opciones) - 1)]
        # Los enteros se decodifican en [bajo - 0,5, alto + 0,5] para que cada entero tenga la misma
        # masa (incluidos los extremos), igual que `suggest_int` de Optuna
        lo, hi = (self.bajo - 0.5, self.alto + 0.5) if self.tipo == "int" else (self.bajo, self.alto)
        if self.log:
            v = float(np.exp(np.log(lo) + g * (np.log(hi) - np.log(lo))))
        else:
            v = lo + g * (hi - lo)
        if self.tipo == "int":
            return int(np.clip(round(v), self.bajo, self.alto))
        return float(np.clip(v, self.bajo, self.alto))


def niveles_log(bajo: float, alto: float, n: int) -> tuple:
    return tuple(float(v) for v in np.geomspace(bajo, alto, n))


@dataclass
class EspecModelo:
    nombre: str
    tarea: str                 # "clf" o "reg"
    prep: str                  # "lineal" o "arboles"
    espacio: list
    fijos: dict = field(default_factory=dict)
    complejidad_fit: str = ""
    complejidad_pred: str = ""

    def tamano_rejilla(self) -> int:
        return int(np.prod([len(p.rejilla) for p in self.espacio]))


# ----------------------------------------------------------------------------------------------
# k-NN con votos ponderados por clase (class_weight no existe en KNeighborsClassifier)
# ----------------------------------------------------------------------------------------------
class KNNPonderado(BaseEstimator, ClassifierMixin):
    """k-NN cuyo voto por clase se multiplica por el peso 'balanced' de esa clase.

    La probabilidad ajustada es p1' = w1 p1 / (w1 p1 + w0 p0), una transformación monótona de la
    fracción de vecinos positivos: cambia el umbral efectivo de decisión, no el orden de los
    puntajes (el AUC es idéntico al del k-NN sin ponderar).
    """

    def __init__(self, n_neighbors=50, weights="uniform", p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        frec = np.bincount(y, minlength=2) / len(y)
        self.peso_ = 0.5 / frec
        self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights,
                                         p=self.p, algorithm="brute", n_jobs=1).fit(X, y)
        return self

    def predict_proba(self, X):
        q = self.knn_.predict_proba(X) * self.peso_
        return q / q.sum(axis=1, keepdims=True)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ----------------------------------------------------------------------------------------------
# Catálogo
# ----------------------------------------------------------------------------------------------
def _knn_espacio():
    # La distancia es euclidiana (p = 2): con fuerza bruta se calcula con productos de matrices
    # (BLAS); la de Manhattan no admite ese atajo y resultó unas 20 veces más lenta en la prueba
    # de tiempos, sin una razón sustantiva para preferirla con variables estandarizadas.
    return [Param("n_neighbors", "int", 5, 600, log=True,
                  rejilla=tuple(int(round(v)) for v in np.geomspace(8, 600, 12))),
            Param("weights", "cat", opciones=("uniform", "distance"), rejilla=("uniform", "distance"))]


def _rf_espacio():
    # max_features <= 0,5: con fracciones mayores los árboles se parecen demasiado entre sí (el
    # bosque pierde la decorrelación que lo justifica) y el costo crece linealmente; max_samples
    # <= 0,7 añade diversidad y reduce el costo por árbol. Ambos límites salen de la prueba de
    # tiempos (las configuraciones extremas tardaban más de 60 s por evaluación).
    return [Param("max_depth", "int", 4, 32, rejilla=(8, 16, 32)),
            Param("min_samples_leaf", "int", 1, 500, log=True, rejilla=(3, 50)),
            Param("max_features", "float", 0.05, 0.5, rejilla=(0.12, 0.35)),
            Param("max_samples", "float", 0.1, 0.7, rejilla=(0.25, 0.6))]


def _xgb_espacio():
    return [Param("learning_rate", "float", 0.01, 0.3, log=True, rejilla=(0.03, 0.1)),
            Param("max_depth", "int", 2, 10, rejilla=(3, 6, 9)),
            Param("min_child_weight", "float", 1.0, 300.0, log=True, rejilla=(3.0, 60.0)),
            Param("colsample_bytree", "float", 0.3, 1.0, rejilla=(0.5, 0.9)),
            Param("subsample", "float", 0.5, 1.0, rejilla=(0.8,))]


def catalogo(tarea: str) -> dict[str, EspecModelo]:
    """Los siete modelos de la Etapa 1 para la tarea indicada ("clf" o "reg")."""
    if tarea == "clf":
        modelos = [
            EspecModelo("KNN", "clf", "lineal", _knn_espacio(),
                        complejidad_fit="O(1) (almacena n×p)", complejidad_pred="O(n·p) por consulta (fuerza bruta)"),
            EspecModelo("Naive Bayes", "clf", "lineal",
                        [Param("var_smoothing", "float", 1e-12, 1.0, log=True, rejilla=niveles_log(1e-12, 1.0, 24))],
                        complejidad_fit="O(n·p)", complejidad_pred="O(p) por consulta"),
            EspecModelo("Regresión logística", "clf", "lineal",
                        [Param("C", "float", 1e-4, 1e2, log=True, rejilla=niveles_log(1e-4, 1e2, 12)),
                         Param("l1_ratio", "cat", opciones=(0.0, 1.0), rejilla=(0.0, 1.0))],
                        complejidad_fit="O(n·p) por época (SAGA)", complejidad_pred="O(p)"),
            EspecModelo("Decision Tree", "clf", "arboles",
                        [Param("max_depth", "int", 3, 30, rejilla=(4, 8, 16)),
                         Param("min_samples_leaf", "int", 5, 5000, log=True, rejilla=(20, 500)),
                         Param("criterion", "cat", opciones=("gini", "entropy"), rejilla=("gini", "entropy")),
                         Param("max_features", "float", 0.2, 1.0, rejilla=(0.5, 1.0))],
                        complejidad_fit="O(p·n log n)", complejidad_pred="O(profundidad)"),
            EspecModelo("Random Forest", "clf", "arboles", _rf_espacio(), {"n_estimators": N_ARBOLES_RF},
                        complejidad_fit="O(T·m·n' log n')", complejidad_pred="O(T·profundidad)"),
            EspecModelo("XGBoost", "clf", "arboles", _xgb_espacio(), {"n_estimators": N_ARBOLES_XGB},
                        complejidad_fit="O(T·(n·p + 2^d·p·b)) con histogramas", complejidad_pred="O(T·d)"),
            EspecModelo("SVM", "clf", "lineal",
                        [Param("gamma", "float", 1e-4, 1.0, log=True, rejilla=niveles_log(3e-4, 1e-1, 6)),
                         Param("C", "float", 1e-3, 10.0, log=True, rejilla=niveles_log(1e-3, 1.0, 4))],
                        {"n_components": N_COMPONENTES_NYSTROEM},
                        complejidad_fit="O(n·m·p + n·m²) (Nyström m=300 + LinearSVC)", complejidad_pred="O(m·p)"),
        ]
    else:
        modelos = [
            EspecModelo("KNN", "reg", "lineal", _knn_espacio(),
                        complejidad_fit="O(1)", complejidad_pred="O(n·p) por consulta"),
            EspecModelo("Ridge", "reg", "lineal",
                        [Param("alpha", "float", 1e-3, 1e5, log=True, rejilla=niveles_log(1e-3, 1e5, 24))],
                        complejidad_fit="O(n·p² + p³)", complejidad_pred="O(p)"),
            EspecModelo("Lasso", "reg", "lineal",
                        [Param("alpha", "float", 1e-6, 1e-1, log=True, rejilla=niveles_log(1e-6, 1e-1, 24))],
                        complejidad_fit="O(n·p) por ciclo (descenso por coordenadas)", complejidad_pred="O(p)"),
            EspecModelo("Decision Tree", "reg", "arboles",
                        [Param("max_depth", "int", 3, 30, rejilla=(4, 8, 16)),
                         Param("min_samples_leaf", "int", 5, 5000, log=True, rejilla=(20, 500)),
                         Param("ccp_alpha", "float", 1e-7, 1e-2, log=True, rejilla=(1e-6, 1e-4)),
                         Param("max_features", "float", 0.2, 1.0, rejilla=(0.5, 1.0))],
                        complejidad_fit="O(p·n log n)", complejidad_pred="O(profundidad)"),
            EspecModelo("Random Forest", "reg", "arboles", _rf_espacio(), {"n_estimators": N_ARBOLES_RF},
                        complejidad_fit="O(T·m·n' log n')", complejidad_pred="O(T·profundidad)"),
            EspecModelo("XGBoost", "reg", "arboles", _xgb_espacio(), {"n_estimators": N_ARBOLES_XGB},
                        complejidad_fit="O(T·(n·p + 2^d·p·b))", complejidad_pred="O(T·d)"),
            EspecModelo("SVR", "reg", "lineal",
                        [Param("gamma", "float", 1e-4, 1.0, log=True, rejilla=niveles_log(1e-3, 1e-1, 3)),
                         Param("C", "float", 1e-3, 10.0, log=True, rejilla=niveles_log(1e-2, 10.0, 4)),
                         Param("epsilon", "float", 0.01, 0.5, log=True, rejilla=(0.05, 0.2))],
                        {"n_components": N_COMPONENTES_NYSTROEM},
                        complejidad_fit="O(n·m·p + n·m²) (Nyström + LinearSVR)", complejidad_pred="O(m·p)"),
        ]
    return {m.nombre: m for m in modelos}


def construir(espec: EspecModelo, params: dict, balanceo: str = "Sin balanceo",
              y=None, semilla: int = SEMILLA):
    """Crea el estimador del modelo `espec` con los hiperparámetros `params`.

    `balanceo == "class_weight"` se traduce a la forma nativa de cada modelo:
    `class_weight="balanced"` (logística, árboles, Random Forest, LinearSVC),
    `scale_pos_weight = n_neg / n_pos` (XGBoost), priores uniformes (Naive Bayes, equivalente a
    ponderar las clases) y votos ponderados (k-NN). SMOTE y ADASYN no se aplican aquí: remuestrean
    los datos de entrenamiento de cada pliegue antes del ajuste (ver `remuestrear`).
    """
    kw = {**espec.fijos, **params}
    cw = balanceo == "class_weight"
    n = espec.nombre
    if espec.tarea == "clf":
        if n == "KNN":
            if cw:
                return KNNPonderado(**kw)
            return KNeighborsClassifier(algorithm="brute", n_jobs=1, **kw)
        if n == "Naive Bayes":
            return GaussianNB(priors=[0.5, 0.5] if cw else None, **kw)
        if n == "Regresión logística":
            return LogisticRegression(solver="saga", max_iter=300, tol=1e-3, random_state=semilla,
                                      class_weight="balanced" if cw else None, **kw)
        if n == "Decision Tree":
            return DecisionTreeClassifier(random_state=semilla, class_weight="balanced" if cw else None, **kw)
        if n == "Random Forest":
            return RandomForestClassifier(n_jobs=1, random_state=semilla,
                                          class_weight="balanced" if cw else None, **kw)
        if n == "XGBoost":
            spw = 1.0
            if cw:
                y = np.asarray(y)
                spw = float((y == 0).sum() / max((y == 1).sum(), 1))
            return XGBClassifier(tree_method="hist", n_jobs=1, random_state=semilla, verbosity=0,
                                 scale_pos_weight=spw, **kw)
        if n == "SVM":
            return Pipeline([("nys", Nystroem(kernel="rbf", gamma=kw["gamma"], n_components=kw["n_components"],
                                              random_state=semilla)),
                             ("svc", LinearSVC(C=kw["C"], class_weight="balanced" if cw else None,
                                               max_iter=5000, random_state=semilla))])
    else:
        if n == "KNN":
            return KNeighborsRegressor(algorithm="brute", n_jobs=1, **kw)
        if n == "Ridge":
            return Ridge(random_state=semilla, **kw)
        if n == "Lasso":
            return Lasso(max_iter=5000, tol=1e-4, random_state=semilla, **kw)
        if n == "Decision Tree":
            return DecisionTreeRegressor(random_state=semilla, **kw)
        if n == "Random Forest":
            return RandomForestRegressor(n_jobs=1, random_state=semilla, **kw)
        if n == "XGBoost":
            return XGBRegressor(tree_method="hist", n_jobs=1, random_state=semilla, verbosity=0, **kw)
        if n == "SVR":
            return Pipeline([("nys", Nystroem(kernel="rbf", gamma=kw["gamma"], n_components=kw["n_components"],
                                              random_state=semilla)),
                             ("svr", LinearSVR(C=kw["C"], epsilon=kw["epsilon"], max_iter=5000,
                                               random_state=semilla))])
    raise ValueError(f"Modelo desconocido: {n} ({espec.tarea})")


def remuestrear(X, y, balanceo: str, semilla: int = SEMILLA):
    """Aplica SMOTE o ADASYN a los datos de entrenamiento ya preprocesados de un pliegue.

    Solo se llama con datos de entrenamiento: la validación nunca se remuestrea. Las demás
    técnicas devuelven los datos sin cambios.
    """
    if balanceo == "SMOTE":
        return SMOTE(k_neighbors=5, random_state=semilla).fit_resample(X, y)
    if balanceo == "ADASYN":
        return ADASYN(n_neighbors=5, random_state=semilla).fit_resample(X, y)
    return X, y


def puntaje(modelo, X) -> np.ndarray:
    """Puntaje continuo para ordenar (probabilidad de la clase 1 o función de decisión)."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def es_probabilistico(espec: EspecModelo) -> bool:
    return espec.tarea == "clf" and espec.nombre != "SVM"
