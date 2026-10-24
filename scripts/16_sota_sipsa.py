"""Comparadores recientes del estado del arte en el problema SIPSA: LightGBM y CatBoost.

Mismo protocolo que los 140 modelos del Entregable 2 y que el EJUL: pliegues externos k1-k4 y
final, muestra S_k de 40 mil filas, tres pliegues internos temporales, preprocesamiento de árboles
ajustado dentro de cada pliegue, TPE de Optuna con semilla 42 y 24 evaluaciones (8 de arranque),
criterio AUC interno, umbral por máximo F1 en las predicciones internas y una única evaluación en V.

Uso: python scripts/16_sota_sipsa.py LightGBM k1 [n_jobs]
"""
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config as C  # noqa: E402
import datos as D  # noqa: E402
import evaluacion as EV  # noqa: E402
import experimento as X  # noqa: E402
import modelos as M  # noqa: E402
import optimizacion as O  # noqa: E402
import preprocesamiento as P  # noqa: E402

SALIDA = RAIZ / "data" / "resultados" / "sota"
ESPACIOS = {
    "LightGBM": [M.Param("learning_rate", "float", 0.01, 0.3, log=True),
                 M.Param("num_leaves", "int", 8, 256, log=True),
                 M.Param("min_child_samples", "int", 5, 500, log=True),
                 M.Param("colsample_bytree", "float", 0.3, 1.0),
                 M.Param("subsample", "float", 0.5, 1.0),
                 M.Param("reg_lambda", "float", 1e-3, 10.0, log=True)],
    "CatBoost": [M.Param("learning_rate", "float", 0.01, 0.3, log=True),
                 M.Param("depth", "int", 3, 10),
                 M.Param("l2_leaf_reg", "float", 1.0, 30.0, log=True),
                 M.Param("bagging_temperature", "float", 0.0, 1.0),
                 M.Param("random_strength", "float", 0.1, 10.0, log=True)],
}


def construir(nombre, params, n_jobs, semilla=C.SEMILLA):
    if nombre == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, subsample_freq=1, random_state=semilla, n_jobs=n_jobs, verbose=-1, **params)
    if nombre == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, random_seed=semilla, thread_count=n_jobs, verbose=0,
                                  allow_writing_files=False, **params)
    raise ValueError(nombre)


class Objetivo:
    def __init__(self, nombre, internos, n_jobs):
        self.nombre, self.internos, self.n_jobs = nombre, internos, n_jobs
        self.cache, self.predicciones = {}, {}

    def __call__(self, params):
        clave = O._clave(params)
        if clave in self.cache:
            return {**self.cache[clave], "repetida": True, "segundos": 0.0}
        t0 = time.perf_counter()
        aucs, preds = [], []
        for X_tr, y_tr, X_va, y_va in self.internos:
            m = construir(self.nombre, params, self.n_jobs).fit(X_tr, y_tr)
            p = m.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, p))
            preds.append(p.astype(np.float32))
        r = {"puntaje": float(np.mean(aucs)), "puntajes_pliegue": aucs, "segundos": time.perf_counter() - t0, "repetida": False}
        self.cache[clave] = {k: v for k, v in r.items() if k not in ("segundos", "repetida")}
        self.predicciones[clave] = preds
        return r


def ejecutar(nombre, pliegue, n_jobs=2):
    archivo = SALIDA / f"{nombre}_{pliegue}.joblib"
    if archivo.exists():
        return joblib.load(archivo)
    warnings.filterwarnings("ignore")
    C.fijar_semillas(C.SEMILLA)
    t_ini = time.perf_counter()
    S, V = X.cargar_pliegue(pliegue)
    internos, y_int = [], []
    for tr, va in D.pliegues_internos(S["semana_t1"]):
        Str, Sva = S.iloc[tr], S.iloc[va]
        prep = P.preprocesador("arboles", "clf")
        internos.append((prep.fit_transform(Str, Str[C.OBJ_CLF].to_numpy()), Str[C.OBJ_CLF].to_numpy(),
                         prep.transform(Sva), Sva[C.OBJ_CLF].to_numpy()))
        y_int.append(Sva[C.OBJ_CLF].to_numpy())
    obj = Objetivo(nombre, internos, n_jobs)
    t0 = time.perf_counter()
    res = O.bayes_tpe(obj, ESPACIOS[nombre], C.PRESUPUESTO, C.SEMILLA)
    t_busqueda = time.perf_counter() - t0
    hist = pd.DataFrame(res["historial"])
    i = int(hist["puntaje"].to_numpy().argmax())
    mejor = hist.loc[i, "params"]
    umbral = EV.umbral_f1(np.concatenate(y_int), np.concatenate(obj.predicciones[O._clave(mejor)]).astype(float))
    prep = P.preprocesador("arboles", "clf")
    t0 = time.perf_counter()
    X_S = prep.fit_transform(S, S[C.OBJ_CLF].to_numpy())
    modelo = construir(nombre, mejor, n_jobs).fit(X_S, S[C.OBJ_CLF].to_numpy())
    t_refit = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_V = modelo.predict_proba(prep.transform(V))[:, 1]
    t_pred = time.perf_counter() - t0
    resumen = {"tarea": "clf", "modelo": nombre, "balanceo": "Sin balanceo", "optimizador": "Bayesiana (TPE)",
               "pliegue": pliegue, "semilla": C.SEMILLA, "presupuesto": C.PRESUPUESTO, "evaluaciones": len(hist),
               "params": mejor, "puntaje_interno": float(hist.loc[i, "puntaje"]), "t_busqueda": t_busqueda,
               "t_por_evaluacion": float(hist["segundos"].mean()), "t_refit": t_refit, "t_pred": t_pred,
               "t_total": time.perf_counter() - t_ini, "filas_entrenamiento": int(len(S)), "filas_eval": int(len(V)),
               "n_variables": int(X_S.shape[1])}
    resumen.update(EV.metricas_clf(V[C.OBJ_CLF].to_numpy(), p_V, umbral, p_V))
    out = {"resumen": resumen, "historial": hist, "pred": {"fila": V["fila"].to_numpy(), "s": p_V.astype(np.float32)}}
    SALIDA.mkdir(parents=True, exist_ok=True)
    tmp = archivo.with_suffix(".tmp")
    joblib.dump(out, tmp, compress=3)
    os.replace(tmp, archivo)
    return out


if __name__ == "__main__":
    nombre, pl = sys.argv[1], sys.argv[2]
    nj = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    r = ejecutar(nombre, pl, nj)["resumen"]
    print(nombre, pl, {k: round(r[k], 4) for k in ["auc", "auc_pr", "brier", "log_loss", "ece", "f1"]}, r["params"],
          f"{r['t_total']:.0f}s", flush=True)
