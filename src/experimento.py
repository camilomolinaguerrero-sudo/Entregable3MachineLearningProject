"""Validación anidada del pipeline combinatorio (modelo → balanceo → optimizador).

Un "trabajo" es una combinación (tarea, modelo, balanceo, optimizador) evaluada en un pliegue
externo. Para cada trabajo:

1. Se toma la muestra de entrenamiento externa S (40 mil filas estratificadas por semana) y el
   bloque de validación externo V completo (52 semanas, ~100 mil filas).
2. Bucle interno: tres pliegues de ventana creciente sobre las semanas de S (hueco de 8 semanas).
   En cada uno se ajusta el preprocesador solo con su entrenamiento, se remuestrea solo ese
   entrenamiento y se evalúan las configuraciones que propone el optimizador.
3. La mejor configuración se reajusta con todo S (preprocesador y remuestreo reajustados en S) y
   se evalúa UNA vez en V. Esa es la métrica que se reporta (nunca el óptimo interno).
4. El umbral de decisión (máximo F1) y, para la SVM, el escalado de Platt se estiman con las
   predicciones internas fuera de pliegue de la mejor configuración, nunca con V.

El pliegue "final" repite el procedimiento con una muestra de todo el periodo de entrenamiento y
evalúa en la prueba de 2025; produce los modelos que se interpretan y comparan en la prueba.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config as C
import datos as D
import evaluacion as EV
import modelos as M
import optimizacion as O
import preprocesamiento as P

PLIEGUES = ["k1", "k2", "k3", "k4", "final"]
DIR_PLIEGUES = C.INTERIM / "pliegues"
COLS_GUARDAR = sorted(set(C.X_COLS + [C.OBJ_CLF, C.OBJ_REG, "semana_t1", "fila"]))


def preparar_pliegues(n_muestra: int = C.N_MUESTRA, semilla: int = C.SEMILLA) -> pd.DataFrame:
    """Guarda en disco S (muestra de entrenamiento) y V (evaluación) de cada pliegue externo.

    Returns
    -------
    Tabla descriptiva de los pliegues (fechas y tamaños).
    """
    DIR_PLIEGUES.mkdir(parents=True, exist_ok=True)
    p = D.cargar_panel()
    train, test = D.particion(p)
    filas = []
    externos = D.pliegues_externos(train)
    for i, nombre in enumerate(PLIEGUES):
        if nombre == "final":
            T, V = train, test
        else:
            a, b = externos[i]
            T, V = train.iloc[a], train.iloc[b]
        S = D.muestra_por_semana(T, n_muestra, semilla + i)
        S[COLS_GUARDAR].to_parquet(DIR_PLIEGUES / f"{nombre}_S.parquet", index=False)
        V[COLS_GUARDAR].to_parquet(DIR_PLIEGUES / f"{nombre}_V.parquet", index=False)
        filas.append({"pliegue": nombre, "train_desde": T.semana_t1.min(), "train_hasta": T.semana_t1.max(),
                      "filas_train": len(T), "filas_muestra": len(S), "eval_desde": V.semana_t1.min(),
                      "eval_hasta": V.semana_t1.max(), "filas_eval": len(V),
                      "prev_muestra": S[C.OBJ_CLF].mean(), "prev_eval": V[C.OBJ_CLF].mean()})
    tabla = pd.DataFrame(filas)
    tabla.to_parquet(DIR_PLIEGUES / "descripcion.parquet", index=False)
    return tabla


def cargar_pliegue(nombre: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_parquet(DIR_PLIEGUES / f"{nombre}_S.parquet"),
            pd.read_parquet(DIR_PLIEGUES / f"{nombre}_V.parquet"))


def _objetivo(tarea):
    return C.OBJ_CLF if tarea == "clf" else C.OBJ_REG


def cache_interna(espec, S: pd.DataFrame, balanceo: str, semilla: int = C.SEMILLA):
    """Preprocesa (y remuestrea) los pliegues internos de S una sola vez.

    Returns
    -------
    lista de (X_tr, y_tr, X_va, y_va) y lista de posiciones de validación interna.
    """
    y_col = _objetivo(espec.tarea)
    pliegues, posiciones = [], []
    for tr, va in D.pliegues_internos(S["semana_t1"]):
        Str, Sva = S.iloc[tr], S.iloc[va]
        prep = P.preprocesador(espec.prep, espec.tarea)
        X_tr = prep.fit_transform(Str, Str[y_col].to_numpy())
        X_va = prep.transform(Sva)
        y_tr, y_va = Str[y_col].to_numpy(), Sva[y_col].to_numpy()
        if espec.tarea == "clf":
            X_tr, y_tr = M.remuestrear(X_tr, y_tr, balanceo, semilla)
        pliegues.append((X_tr, y_tr, X_va, y_va))
        posiciones.append(va)
    return pliegues, posiciones


def nombre_trabajo(tarea, modelo, balanceo, optimizador, pliegue) -> str:
    limpio = lambda s: "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")
    return f"{tarea}__{limpio(modelo)}__{limpio(balanceo)}__{limpio(optimizador)}__{pliegue}"


def ejecutar_trabajo(tarea: str, modelo: str, balanceo: str, optimizador: str, pliegue: str,
                     presupuesto: int = C.PRESUPUESTO, semilla: int = C.SEMILLA,
                     guardar: bool = True, dir_salida: Path | None = None) -> dict:
    """Corre una combinación en un pliegue externo y guarda el resultado (con punto de control).

    Returns
    -------
    dict con `resumen` (una fila de la tabla maestra), `historial` (todas las evaluaciones),
    `generaciones` (solo AG) y `pred` (puntajes en V, alineados con la columna `fila`).
    """
    dir_salida = dir_salida or (C.RES / "trabajos")
    archivo = dir_salida / f"{nombre_trabajo(tarea, modelo, balanceo, optimizador, pliegue)}.joblib"
    if guardar and archivo.exists():
        return joblib.load(archivo)
    warnings.filterwarnings("ignore")
    C.fijar_semillas(semilla)
    espec = M.catalogo(tarea)[modelo]
    S, V = cargar_pliegue(pliegue)
    y_col = _objetivo(tarea)
    t_ini = time.perf_counter()

    # Bucle interno
    t0 = time.perf_counter()
    pliegues, pos_internas = cache_interna(espec, S, balanceo, semilla)
    t_cache = time.perf_counter() - t0
    obj = O.Objetivo(espec, pliegues, balanceo, semilla)
    t0 = time.perf_counter()
    res = O.OPTIMIZADORES[optimizador](obj, espec.espacio, presupuesto, semilla)
    t_busqueda = time.perf_counter() - t0
    hist = pd.DataFrame(res["historial"])
    i_mejor = int(hist["puntaje"].to_numpy().argmax())
    mejor = hist.loc[i_mejor, "params"]

    # Predicciones internas fuera de pliegue de la mejor configuración -> umbral y Platt
    preds_int = obj.predicciones[O._clave(mejor)]
    y_int = np.concatenate([p[3] for p in pliegues])
    s_int = np.concatenate(preds_int).astype(float)

    # Reajuste con todo S y evaluación única en V
    prep = P.preprocesador(espec.prep, tarea)
    t0 = time.perf_counter()
    X_S = prep.fit_transform(S, S[y_col].to_numpy())
    y_S = S[y_col].to_numpy()
    if tarea == "clf":
        X_S, y_S = M.remuestrear(X_S, y_S, balanceo, semilla)
    modelo_final = M.construir(espec, mejor, balanceo, y_S, semilla)
    modelo_final.fit(X_S, y_S)
    t_refit = time.perf_counter() - t0
    t0 = time.perf_counter()
    X_V = prep.transform(V)
    if tarea == "clf":
        s_V = M.puntaje(modelo_final, X_V)
    else:
        s_V = modelo_final.predict(X_V)
    t_pred = time.perf_counter() - t0

    resumen = {"tarea": tarea, "modelo": modelo, "balanceo": balanceo, "optimizador": optimizador,
               "pliegue": pliegue, "semilla": semilla, "presupuesto": presupuesto,
               "evaluaciones": len(hist), "params": mejor, "puntaje_interno": float(hist.loc[i_mejor, "puntaje"]),
               "t_cache": t_cache, "t_busqueda": t_busqueda, "t_por_evaluacion": float(hist["segundos"].mean()),
               "t_refit": t_refit, "t_pred": t_pred, "t_total": time.perf_counter() - t_ini,
               "filas_entrenamiento": int(len(y_S)), "filas_eval": int(len(V)),
               "n_variables": int(X_S.shape[1])}
    pred = {"fila": V["fila"].to_numpy(), "s": s_V.astype(np.float32)}
    if tarea == "clf":
        umbral = EV.umbral_f1(y_int, s_int)
        if M.es_probabilistico(espec):
            prob_V = s_V
        else:
            platt = EV.Platt().fit(s_int, y_int)
            prob_V = platt.predict(s_V)
            pred["prob"] = prob_V.astype(np.float32)
        resumen.update(EV.metricas_clf(V[y_col].to_numpy(), s_V, umbral, prob_V))
    else:
        resumen.update(EV.metricas_reg(V[y_col].to_numpy(), s_V, V[C.OBJ_CLF].to_numpy()))
        resumen["rmse_interno"] = -resumen["puntaje_interno"]
    salida = {"resumen": resumen, "historial": hist, "generaciones": res.get("generaciones"), "pred": pred}
    if guardar:
        dir_salida.mkdir(parents=True, exist_ok=True)
        tmp = archivo.with_suffix(".tmp")
        joblib.dump(salida, tmp, compress=3)
        os.replace(tmp, archivo)
    return salida


def lista_trabajos(tareas=("clf", "reg"), pliegues=PLIEGUES) -> list[tuple]:
    """Todas las combinaciones del pipeline: 7 × 4 × 4 en clasificación y 7 × 4 en regresión."""
    trabajos = []
    for tarea in tareas:
        balanceos = M.BALANCEOS if tarea == "clf" else ["Sin balanceo"]
        for modelo in M.catalogo(tarea):
            for balanceo in balanceos:
                for opt in M.OPTIMIZADORES:
                    for pl in pliegues:
                        trabajos.append((tarea, modelo, balanceo, opt, pl))
    return trabajos
