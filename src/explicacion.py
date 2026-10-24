"""Explicación local por valores de Shapley estimados con permutaciones (Štrumbelj y Kononenko, 2014).

Se usa en el dashboard porque funciona con cualquier tubería que reciba el DataFrame crudo
(variables numéricas y categóricas), incluido el modelo original. Para una observación x y un
conjunto de fondo Z:

    φ_j ≈ (1/M) Σ_m [ f(x_{S_m ∪ {j}}, z_m) − f(x_{S_m}, z_m) ],

donde S_m son las variables que preceden a j en la permutación aleatoria π_m y z_m es una fila de
fondo elegida al azar. Cada permutación cumple exactamente la eficiencia,
Σ_j φ_j = f(x) − f(z_m), de modo que la suma de las contribuciones es f(x) − promedio de f(z_m).
Todas las filas híbridas se evalúan en un único lote de M (p + 1) filas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def shap_permutaciones(predecir, x: pd.DataFrame, fondo: pd.DataFrame, M: int = 64, semilla: int = 42) -> pd.Series:
    """Valores de Shapley (escala de probabilidad) de la fila `x` (DataFrame de una fila).

    Parameters
    ----------
    predecir : función DataFrame → vector de probabilidades.
    x : observación a explicar (1 fila, mismas columnas que `fondo`).
    fondo : filas de referencia (distribución de comparación).
    M : número de permutaciones.
    """
    rng = np.random.default_rng(semilla)
    cols = list(x.columns)
    p = len(cols)
    filas, orden = [], []
    xv = x.iloc[0]
    for _ in range(M):
        pi = rng.permutation(p)
        z = fondo.iloc[rng.integers(len(fondo))].copy()
        filas.append(z.copy())
        for j in pi:
            z[cols[j]] = xv[cols[j]]
            filas.append(z.copy())
        orden.append(pi)
    lote = pd.DataFrame(filas, columns=cols)
    for c in cols:                                   # conserva los tipos originales
        lote[c] = lote[c].astype(fondo[c].dtype, errors="ignore")
    f = np.asarray(predecir(lote), float).reshape(M, p + 1)
    phi = np.zeros(p)
    for m, pi in enumerate(orden):
        phi[pi] += np.diff(f[m])
    return pd.Series(phi / M, index=cols)


def shap_global(predecir, X: pd.DataFrame, fondo: pd.DataFrame, M: int = 16, semilla: int = 42) -> pd.DataFrame:
    """Matriz n × p de valores de Shapley para varias filas (para el resumen global)."""
    salida = [shap_permutaciones(predecir, X.iloc[[i]], fondo, M, semilla + i) for i in range(len(X))]
    return pd.DataFrame(salida, index=X.index)
