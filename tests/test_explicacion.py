"""Pruebas de la explicación local por permutaciones."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from explicacion import shap_permutaciones  # noqa: E402


def test_modelo_lineal_recupera_contribuciones_exactas():
    # f(x) = 2 a − b + 0,5 [c == "s"]: el Shapley de cada variable es su efecto frente al fondo
    rng = np.random.default_rng(0)
    fondo = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50), "c": rng.choice(["s", "n"], 50)})
    f = lambda d: 2 * d["a"].to_numpy() - d["b"].to_numpy() + 0.5 * (d["c"] == "s").to_numpy()
    x = pd.DataFrame({"a": [1.0], "b": [2.0], "c": ["s"]})
    phi = shap_permutaciones(f, x, fondo, M=400)
    # eficiencia exacta respecto de las filas de fondo usadas: suma ≈ f(x) − E[f(Z)]
    assert abs(phi.sum() - (f(x)[0] - f(fondo).mean())) < 0.15
    assert abs(phi["a"] - 2 * (1.0 - fondo["a"].mean())) < 0.25
    assert abs(phi["b"] - (-(2.0 - fondo["b"].mean()))) < 0.25


def test_variable_irrelevante_tiene_shapley_cero():
    fondo = pd.DataFrame({"a": np.linspace(-1, 1, 20), "b": np.linspace(5, 6, 20)})
    f = lambda d: d["a"].to_numpy() ** 2
    phi = shap_permutaciones(f, pd.DataFrame({"a": [0.7], "b": [100.0]}), fondo, M=50)
    assert abs(phi["b"]) < 1e-12
