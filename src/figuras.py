"""Estilo y utilidades de figuras compartidas por los capítulos del libro."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paleta de Okabe-Ito (apta para daltonismo) y extensiones
COLOR_MODELO = {
    "KNN": "#E69F00", "Naive Bayes": "#56B4E9", "Regresión logística": "#009E73",
    "Decision Tree": "#CC79A7", "Random Forest": "#0072B2", "XGBoost": "#D55E00", "SVM": "#7F7F7F",
    "Ridge": "#009E73", "Lasso": "#8DD3A0", "SVR": "#7F7F7F",
}
COLOR_OPT = {"Grid Search": "#7F7F7F", "Random Search": "#56B4E9", "Bayesiana (TPE)": "#D55E00",
             "Genética (DEAP)": "#009E73", "Successive Halving": "#CC79A7", "Hyperband": "#0072B2",
             "Random Search (9)": "#56B4E9", "Bayesiana TPE (9)": "#D55E00"}
COLOR_BAL = {"Sin balanceo": "#7F7F7F", "SMOTE": "#E69F00", "ADASYN": "#CC79A7", "class_weight": "#0072B2"}
ORDEN_OPT = ["Grid Search", "Random Search", "Bayesiana (TPE)", "Genética (DEAP)"]
ORDEN_BAL = ["Sin balanceo", "SMOTE", "ADASYN", "class_weight"]
ORDEN_CLF = ["KNN", "Naive Bayes", "Regresión logística", "Decision Tree", "Random Forest", "XGBoost", "SVM"]
ORDEN_REG = ["KNN", "Ridge", "Lasso", "Decision Tree", "Random Forest", "XGBoost", "SVR"]


def estilo():
    plt.rcParams.update({
        "figure.figsize": (9, 4.5), "figure.dpi": 110, "savefig.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "axes.titleweight": "bold", "axes.titlesize": 11, "axes.labelsize": 10, "font.size": 9.5,
        "legend.frameon": False, "legend.fontsize": 8.5,
    })


def coma(x, d=3):
    """Número con coma decimal (convención del informe)."""
    if pd.isna(x):
        return "—"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def tabla_es(df: pd.DataFrame, decimales: int | dict = 3):
    """Formatea un DataFrame con coma decimal para mostrarlo en el libro."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            d = decimales.get(c, 3) if isinstance(decimales, dict) else decimales
            out[c] = out[c].map(lambda v: coma(v, d))
    return out


_POLIGONOS = None


def mapa_base(ax, color="#f1f3f5", borde="#ced4da"):
    """Contorno municipal de Colombia (geojson del Entregable 1) como fondo de los mapas."""
    import json
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
    from config import RAW
    global _POLIGONOS
    if _POLIGONOS is None:
        gj = json.loads((RAW / "municipios_geom.geojson").read_text())
        _POLIGONOS = []
        for f in gj["features"]:
            g = f["geometry"]
            partes = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            _POLIGONOS += [np.asarray(p[0]) for p in partes]
    ax.add_collection(PatchCollection([Polygon(p, closed=True) for p in _POLIGONOS], facecolor=color,
                                      edgecolor=borde, linewidths=0.12))
    ax.set_xlim(-79.5, -66.5)
    ax.set_ylim(-4.5, 13.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def mejor_hasta(puntajes) -> np.ndarray:
    """Curva de desempeño en cualquier momento: el mejor valor encontrado hasta cada evaluación."""
    return np.maximum.accumulate(np.asarray(puntajes, float))
