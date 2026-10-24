"""Autocorrelación espacial de residuos: pesos kNN, I de Moran global y LISA.

Mismas definiciones que el Entregable 1 (k = 8 vecinos con distancia haversine, pesos
estandarizados por filas, prueba de permutación con 999 réplicas) para que los valores sean
comparables con el I = 0,045 que señaló la retroalimentación.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from config import SEMILLA


def pesos_knn(lat, lon, k: int = 8) -> np.ndarray:
    """Matriz de pesos de k vecinos más cercanos (haversine), estandarizada por filas."""
    X = np.radians(np.column_stack([lat, lon]))
    _, idx = BallTree(X, metric="haversine").query(X, k=k + 1)
    idx = idx[:, 1:]
    n = len(lat)
    W = np.zeros((n, n))
    W[np.repeat(np.arange(n), k), idx.ravel()] = 1.0 / k
    return W


def moran_i(y, W, permutaciones: int = 999, semilla: int = SEMILLA) -> dict:
    """I de Moran global con prueba de permutación (p unilateral en la dirección observada).

    Returns
    -------
    dict con I, p, esperanza bajo H0 (-1/(n-1)) y puntaje z de la permutación.
    """
    y = np.asarray(y, float)
    z = y - y.mean()
    n, s0 = len(y), W.sum()
    I = n / s0 * (z @ W @ z) / (z @ z)
    rng = np.random.default_rng(semilla)
    sim = np.empty(permutaciones)
    for i in range(permutaciones):
        zp = rng.permutation(z)
        sim[i] = n / s0 * (zp @ W @ zp) / (zp @ zp)
    if I >= sim.mean():
        p = (np.sum(sim >= I) + 1) / (permutaciones + 1)
    else:
        p = (np.sum(sim <= I) + 1) / (permutaciones + 1)
    return {"I": I, "p": p, "E_I": -1 / (n - 1), "z": (I - sim.mean()) / sim.std()}


def residuo_municipal(df: pd.DataFrame, residuo: str, min_filas: int = 20) -> pd.DataFrame:
    """Residuo medio por municipio de origen (solo municipios con al menos `min_filas` filas)."""
    r = df.groupby("cod_mpio").agg(r=(residuo, "mean"), n=(residuo, "size"),
                                    lat=("lat_o", "first"), lon=("lon_o", "first"))
    return r[r["n"] >= min_filas]


def moran_residuos(df: pd.DataFrame, residuo: str, k: int = 8, min_filas: int = 20,
                   permutaciones: int = 999) -> dict:
    """I de Moran del residuo medio municipal, como en la sección 4.8 del Entregable 1."""
    r = residuo_municipal(df, residuo, min_filas)
    W = pesos_knn(r["lat"].to_numpy(), r["lon"].to_numpy(), k)
    out = moran_i(r["r"].to_numpy(), W, permutaciones)
    out["municipios"] = len(r)
    return out


def lisa(y, W, permutaciones: int = 999, semilla: int = SEMILLA):
    """I de Moran local con p por permutación condicional y cuadrante (Alto-Alto, ...)."""
    y = np.asarray(y, float)
    z = (y - y.mean()) / y.std()
    lag = W @ z
    Ii = z * lag
    rng = np.random.default_rng(semilla)
    n = len(y)
    k = (W > 0).sum(axis=1)
    p = np.empty(n)
    for i in range(n):
        otros = np.delete(z, i)
        sims = np.array([otros[rng.choice(n - 1, k[i], replace=False)].mean()
                         for _ in range(permutaciones)]) * z[i]
        p[i] = (np.sum(np.abs(sims) >= abs(Ii[i])) + 1) / (permutaciones + 1)
    cuadrante = np.where(z > 0, np.where(lag > 0, "Alto-Alto", "Alto-Bajo"),
                         np.where(lag > 0, "Bajo-Alto", "Bajo-Bajo"))
    return Ii, p, cuadrante


def correlograma(df: pd.DataFrame, residuo: str, ks=(4, 8, 16, 32, 64), min_filas: int = 20,
                 permutaciones: int = 499) -> pd.DataFrame:
    """I de Moran del residuo municipal para distintos tamaños de vecindario (escala espacial)."""
    r = residuo_municipal(df, residuo, min_filas)
    filas = []
    for k in ks:
        W = pesos_knn(r["lat"].to_numpy(), r["lon"].to_numpy(), k)
        m = moran_i(r["r"].to_numpy(), W, permutaciones)
        filas.append({"k": k, "I": m["I"], "p": m["p"]})
    return pd.DataFrame(filas)
