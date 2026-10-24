"""Carga del panel, variables nuevas, partición y pliegues de la validación anidada.

El panel de corredores (municipio de origen → central mayorista, por semana) se construyó en el
Entregable 1. Este módulo le añade:

* el objetivo de regresión `y_reg = log(1 + tasa_t1 / base8)`, la versión continua de la
  interrupción (`interrupcion_t1 = 1` exactamente cuando `y_reg < log 1,5`);
* cuatro variables de vecindad espacial que responden a la dependencia espacial que quedó en los
  residuos del modelo base (I de Moran 0,045, p = 0,005).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import BallTree

from config import (FIN_TRAIN, HUECO, INICIO_TEST, N_EXTERNOS, N_INTERNOS, PANEL, RAW,
                    SEMANAS_EXTERNO, SEMILLA, UMBRAL_INTERRUPCION)

K_VECINOS = 8          # municipios vecinos (mismo k que el I de Moran del Entregable 1)
VENTANA_HIST = 26      # semanas para las tasas históricas de vecindad


def vecinos_knn(lat: np.ndarray, lon: np.ndarray, k: int = K_VECINOS) -> np.ndarray:
    """Índices de los k municipios más cercanos (distancia haversine), sin incluir el propio.

    Parameters
    ----------
    lat, lon : arrays de grados decimales, uno por municipio.
    k : número de vecinos.

    Returns
    -------
    ndarray (n, k) con las posiciones de los vecinos de cada municipio.
    """
    X = np.radians(np.column_stack([lat, lon]))
    _, idx = BallTree(X, metric="haversine").query(X, k=k + 1)
    return idx[:, 1:]


def municipios_divipola() -> pd.DataFrame:
    """Código y coordenadas (cabecera municipal) de los 1.122 municipios de la DIVIPOLA."""
    d = pd.read_csv(RAW / "divipola_municipios.csv", dtype=str)
    d["cod_mpio"] = d["cod_mpio"].str.zfill(5)
    d["lat"] = d["latitud"].str.replace(",", ".").astype(float)
    d["lon"] = d["longitud"].str.replace(",", ".").astype(float)
    return d[["cod_mpio", "lat", "lon"]].drop_duplicates("cod_mpio").reset_index(drop=True)


def variables_espaciales(p: pd.DataFrame, k: int = K_VECINOS, ventana: int = VENTANA_HIST) -> pd.DataFrame:
    """Calcula las variables de vecindad con información hasta el cierre de la semana t.

    Una corredor-semana está "en caída" en t si `ratio_t < 0,5` (la misma regla que define la
    interrupción, aplicada a la semana ya observada).

    * `frac_caida_mpio_t`: fracción de los OTROS corredores del mismo municipio de origen (hacia
      otras centrales) en caída en t. Capta choques locales de la oferta.
    * `frac_caida_vec_t`: fracción de corredores de los k municipios vecinos en caída en t (rezago
      espacial de la caída, al estilo de un modelo SLX).
    * `tasa_int_vec26`: tasa de caída de los vecinos en las últimas 26 semanas (t-25..t).
    * `tasa_int_mpio26`: tasa de caída del propio municipio en las últimas 26 semanas.

    Returns
    -------
    DataFrame con las cuatro columnas, alineado con el índice de `p`.
    """
    caida = (p["ratio_t"] < UMBRAL_INTERRUPCION).astype(float)
    clave = pd.DataFrame({"cod_mpio": p["cod_mpio"].to_numpy(), "semana": p["semana"].to_numpy(),
                          "caida": caida.to_numpy()})
    g = clave.groupby(["semana", "cod_mpio"])["caida"].agg(["sum", "count"])
    # El grafo de vecinos se construye con TODOS los municipios de la DIVIPOLA (1.122), un conjunto
    # que no depende del tiempo: si se armara con los municipios que aparecen en el panel, los
    # vecinos de la semana t dependerían de qué municipios entran después a la población en riesgo.
    # Un vecino sin corredores activos en una semana aporta 0 al numerador y al denominador.
    divi = municipios_divipola()
    mpios = divi["cod_mpio"].to_numpy()
    S = g["sum"].unstack(fill_value=0.0).reindex(columns=mpios, fill_value=0.0)
    N = g["count"].unstack(fill_value=0.0).reindex(columns=mpios, fill_value=0.0)
    semanas = pd.date_range(S.index.min(), S.index.max(), freq="W-MON")
    S, N = S.reindex(semanas, fill_value=0.0), N.reindex(semanas, fill_value=0.0)
    vec = vecinos_knn(divi["lat"].to_numpy(), divi["lon"].to_numpy(), k)
    A = np.zeros((len(mpios), len(mpios)))
    A[np.repeat(np.arange(len(mpios)), k), vec.ravel()] = 1.0

    Sv, Nv = S.to_numpy() @ A.T, N.to_numpy() @ A.T          # sumas sobre los vecinos, por semana
    Sr = S.rolling(ventana, min_periods=1).sum().to_numpy()   # ventanas de 26 semanas (incluye t)
    Nr = N.rolling(ventana, min_periods=1).sum().to_numpy()
    Svr, Nvr = Sr @ A.T, Nr @ A.T

    with np.errstate(invalid="ignore", divide="ignore"):
        tablas = {
            "frac_caida_vec_t": Sv / Nv,
            "tasa_int_vec26": Svr / Nvr,
            "tasa_int_mpio26": Sr / Nr,
        }
    fila = pd.Index(semanas).get_indexer(p["semana"])
    col = pd.Index(mpios).get_indexer(p["cod_mpio"])
    out = pd.DataFrame(index=p.index)
    for nombre, M in tablas.items():
        out[nombre] = M[fila, col]
    s_m = S.to_numpy()[fila, col]
    n_m = N.to_numpy()[fila, col]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["frac_caida_mpio_t"] = np.where(n_m > 1, (s_m - caida.to_numpy()) / (n_m - 1), np.nan)
    return out[["frac_caida_mpio_t", "frac_caida_vec_t", "tasa_int_vec26", "tasa_int_mpio26"]]


def cargar_panel() -> pd.DataFrame:
    """Lee el panel del Entregable 1 y añade el objetivo de regresión y las variables espaciales.

    Returns
    -------
    DataFrame ordenado por semana objetivo con una columna `fila` (identificador estable de la
    observación, usado para emparejar predicciones entre modelos).
    """
    p = pd.read_parquet(PANEL)
    p = p.sort_values(["semana_t1", "mercado", "cod_mpio"], kind="mergesort").reset_index(drop=True)
    p["fila"] = np.arange(len(p), dtype=np.int64)
    p["ratio_t1"] = p["tasa_t1"].fillna(0.0) / p["base8"]
    p["y_reg"] = np.log1p(p["ratio_t1"])
    p = pd.concat([p, variables_espaciales(p)], axis=1)
    p["anio_t1"] = p["semana_t1"].dt.year
    p["mes_t1"] = p["semana_t1"].dt.to_period("M").dt.start_time
    return p


def particion(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición cronológica del Entregable 1: (entrenamiento, prueba 2025)."""
    train = p[p["semana_t1"] <= FIN_TRAIN].reset_index(drop=True)
    test = p[p["semana_t1"] >= INICIO_TEST].reset_index(drop=True)
    return train, test


def pliegues_semanales(semanas: pd.Series, n_splits: int, gap: int = HUECO,
                       test_size: int | None = None):
    """Ventana creciente sobre semanas completas (ninguna semana queda partida entre conjuntos).

    Parameters
    ----------
    semanas : Serie con la semana objetivo de cada fila.
    n_splits, gap, test_size : como en `TimeSeriesSplit`, pero contados en semanas.

    Yields
    ------
    (posiciones de entrenamiento, posiciones de validación) sobre las filas de `semanas`.
    """
    s = pd.to_datetime(semanas).to_numpy()
    unicas = np.unique(s)
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)
    for tr_w, va_w in tss.split(unicas):
        yield np.flatnonzero(np.isin(s, unicas[tr_w])), np.flatnonzero(np.isin(s, unicas[va_w]))


def pliegues_externos(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bucle externo: los últimos cuatro bloques de 52 semanas, cada uno con 8 semanas de hueco."""
    return list(pliegues_semanales(train["semana_t1"], N_EXTERNOS, HUECO, SEMANAS_EXTERNO))


def pliegues_internos(semanas: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bucle interno: tres bloques de ventana creciente dentro del entrenamiento externo."""
    return list(pliegues_semanales(semanas, N_INTERNOS, HUECO))


def muestra_por_semana(df: pd.DataFrame, n: int, semilla: int = SEMILLA) -> pd.DataFrame:
    """Muestra aleatoria de `n` filas con la misma fracción en cada semana objetivo.

    Mantiene la composición temporal del bloque (cada semana conserva su peso) para que los
    pliegues internos de ventana creciente sigan teniendo sentido sobre la muestra.
    """
    if n >= len(df):
        return df.copy()
    frac = n / len(df)
    rng = np.random.default_rng(semilla)
    u = rng.random(len(df))
    # Muestreo sistemático dentro de cada semana: se toma la fracción `frac` con menor u.
    rango = df.assign(_u=u).groupby("semana_t1")["_u"].rank(method="first", pct=True)
    out = df.loc[rango <= frac]
    if len(out) > n:
        out = out.sample(n, random_state=semilla)
    return out.sort_values("fila").copy()
