"""Evaluación del AJ-KL en tres conjuntos de datos adicionales con el mismo protocolo.

Para cada conjunto (beijing, bike, bank):
1. El tiempo se divide en 6 tramos consecutivos de igual número de periodos. El pliegue externo j
   (j = 1..5) entrena con los tramos 1..j y evalúa en el tramo j + 1 (ventana creciente, sin barajar).
2. Modelos base: los 7 modelos del curso (sin balanceo) y LightGBM y CatBoost, cada uno ajustado con TPE
   (24 evaluaciones, semilla 42) en tres pliegues internos temporales del entrenamiento, con el AUC como
   criterio, y reentrenado con todo el entrenamiento del pliegue.
3. AJ-KL: el combinador del pliegue j se entrena con las predicciones fuera de muestra de los tramos de
   evaluación anteriores (pliegues 1..j−1); en el pliegue 1, con las predicciones internas de la mejor
   configuración de cada modelo. TPE de 24 evaluaciones sobre sus hiperparámetros, validando en el último
   15 % del historial.
4. Métricas en cada tramo de evaluación y predicciones guardadas para las pruebas estadísticas.
Uso: python scripts/23_externos.py <conjunto> [n_jobs]
"""
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config as C  # noqa: E402
import evaluacion as EV  # noqa: E402
import externos as EX  # noqa: E402
import modelos as M  # noqa: E402
import optimizacion as O  # noqa: E402
from original_model import ApilamientoJerarquicoKL  # noqa: E402

SAL = RAIZ / "data" / "resultados" / "externos"
CLASE = ["XGBoost", "Random Forest", "SVM", "Regresión logística", "KNN", "Decision Tree", "Naive Bayes"]
SOTA = {"LightGBM": [M.Param("learning_rate", "float", 0.01, 0.3, log=True), M.Param("num_leaves", "int", 8, 256, log=True),
                     M.Param("min_child_samples", "int", 5, 500, log=True), M.Param("colsample_bytree", "float", 0.3, 1.0),
                     M.Param("subsample", "float", 0.5, 1.0), M.Param("reg_lambda", "float", 1e-3, 10.0, log=True)],
        "CatBoost": [M.Param("learning_rate", "float", 0.01, 0.3, log=True), M.Param("depth", "int", 3, 10),
                     M.Param("l2_leaf_reg", "float", 1.0, 30.0, log=True), M.Param("bagging_temperature", "float", 0.0, 1.0),
                     M.Param("random_strength", "float", 0.1, 10.0, log=True)]}
ESPACIO_AJKL = [M.Param("lam", "float", 0.01, 1000.0, log=True), M.Param("gamma", "float", 0.95, 1.0),
                M.Param("profundidad", "cat", opciones=(1, 2, 3, 4)), ]


def prep(tipo, num, cat):
    numerico = Pipeline([("imp", SimpleImputer(strategy="median"))] + ([("esc", StandardScaler())] if tipo == "lineal" else []))
    return ColumnTransformer([("num", numerico, num),
                              ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005, sparse_output=False), cat)],
                             sparse_threshold=0.0)


def construir_sota(nombre, params, n_jobs):
    if nombre == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, subsample_freq=1, random_state=C.SEMILLA, n_jobs=n_jobs, verbose=-1, **params)
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=300, random_seed=C.SEMILLA, thread_count=n_jobs, verbose=0, allow_writing_files=False, **params)


class ObjetivoSOTA:
    def __init__(self, nombre, pliegues, n_jobs):
        self.nombre, self.pliegues, self.n_jobs = nombre, pliegues, n_jobs
        self.cache, self.predicciones = {}, {}

    def __call__(self, params):
        clave = O._clave(params)
        if clave in self.cache:
            return {**self.cache[clave], "repetida": True, "segundos": 0.0}
        t0 = time.perf_counter()
        preds = [construir_sota(self.nombre, params, self.n_jobs).fit(a, b).predict_proba(c)[:, 1] for a, b, c, _ in self.pliegues]
        aucs = [roc_auc_score(d, p) for (_, _, _, d), p in zip(self.pliegues, preds)]
        r = {"puntaje": float(np.mean(aucs)), "segundos": time.perf_counter() - t0, "repetida": False}
        self.cache[clave] = {"puntaje": r["puntaje"]}
        self.predicciones[clave] = preds
        return r


def internos(df, y, t, tipo, num, cat):
    """Tres pliegues internos temporales (por periodos) del entrenamiento, preprocesados."""
    periodos = np.unique(t)
    salida, pos = [], []
    for tr_p, va_p in TimeSeriesSplit(n_splits=3).split(periodos):
        tr, va = np.isin(t, periodos[tr_p]), np.isin(t, periodos[va_p])
        pr = prep(tipo, num, cat)
        salida.append((pr.fit_transform(df[tr], y[tr]), y[tr], pr.transform(df[va]), y[va]))
        pos.append(np.flatnonzero(va))
    return salida, pos


def base_pliegue(D, tr, te, n_jobs):
    """Ajusta los 9 modelos base en el entrenamiento del pliegue y predice el tramo de evaluación."""
    df, y, t, num, cat = D["df"], D["y"], D["tiempo"], D["num"], D["cat"]
    salida = {"te": {}, "oof": {}, "oof_pos": None, "params": {}}
    for nombre in CLASE + list(SOTA):
        t0 = time.perf_counter()
        if nombre in CLASE:
            espec = M.catalogo("clf")[nombre]
            pl, pos = internos(df[tr], y[tr], t[tr], espec.prep, num, cat)
            obj = O.Objetivo(espec, pl, "Sin balanceo", C.SEMILLA)
            res = O.bayes_tpe(obj, espec.espacio, C.PRESUPUESTO, C.SEMILLA)
            hist = pd.DataFrame(res["historial"])
            mejor = hist.loc[int(hist["puntaje"].to_numpy().argmax()), "params"]
            pr = prep(espec.prep, num, cat)
            m = M.construir(espec, mejor, "Sin balanceo", y[tr], C.SEMILLA)
            if "n_jobs" in m.get_params() and nombre != "KNN":
                m.set_params(n_jobs=n_jobs)
            m.fit(pr.fit_transform(df[tr], y[tr]), y[tr])
            s_te = M.puntaje(m, pr.transform(df[te]))
            s_oof = np.concatenate(obj.predicciones[O._clave(mejor)]).astype(float)
            if not M.es_probabilistico(espec):
                y_oof = np.concatenate([p[3] for p in pl])
                platt = EV.Platt().fit(s_oof, y_oof)
                s_te, s_oof = platt.predict(s_te), platt.predict(s_oof)
        else:
            pl, pos = internos(df[tr], y[tr], t[tr], "arboles", num, cat)
            obj = ObjetivoSOTA(nombre, pl, n_jobs)
            res = O.bayes_tpe(obj, SOTA[nombre], C.PRESUPUESTO, C.SEMILLA)
            hist = pd.DataFrame(res["historial"])
            mejor = hist.loc[int(hist["puntaje"].to_numpy().argmax()), "params"]
            pr = prep("arboles", num, cat)
            s_te = construir_sota(nombre, mejor, n_jobs).fit(pr.fit_transform(df[tr], y[tr]), y[tr]).predict_proba(pr.transform(df[te]))[:, 1]
            s_oof = np.concatenate(obj.predicciones[O._clave(mejor)]).astype(float)
        salida["te"][nombre] = np.asarray(s_te, float)
        salida["oof"][nombre] = s_oof
        salida["params"][nombre] = mejor
        salida["oof_pos"] = np.flatnonzero(tr)[np.concatenate(pos)]
        print(f"  {nombre:20s} AUC tramo {roc_auc_score(y[te], s_te):.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return salida


def matriz(P, G):
    return np.c_[np.clip(P, 1e-4, 1 - 1e-4), G]


class ObjetivoAJKL:
    def __init__(self, Xh, yh, th, es_val):
        self.Xh, self.yh, self.th, self.va = Xh, yh, th, es_val
        self.cache = {}

    def __call__(self, params):
        clave = O._clave(params)
        if clave in self.cache:
            return {**self.cache[clave], "repetida": True, "segundos": 0.0}
        t0 = time.perf_counter()
        tr, va = ~self.va, self.va
        m = ApilamientoJerarquicoKL(n_modelos=self.Xh.shape[1] - 3, calibrar=True, **params)
        m.fit(self.Xh[tr], self.yh[tr], tiempos=self.th[tr])
        q = m.probabilidad_pool(self.Xh[va])
        p = m.predict_proba(self.Xh[va])[:, 1]
        r = {"puntaje": float(roc_auc_score(self.yh[va], q)), "calibrar": bool(log_loss(self.yh[va], p) <= log_loss(self.yh[va], q)),
             "segundos": time.perf_counter() - t0, "repetida": False}
        self.cache[clave] = {k: v for k, v in r.items() if k not in ("segundos", "repetida")}
        return r


def ejecutar(conjunto, n_jobs=2):
    archivo = SAL / f"{conjunto}.joblib"
    if archivo.exists():
        return joblib.load(archivo)
    warnings.filterwarnings("ignore")
    C.fijar_semillas(C.SEMILLA)
    D = EX.CARGADORES[conjunto]()
    y, t = D["y"], D["tiempo"]
    G = np.column_stack([pd.factorize(D["grupos"][c])[0] for c in ["n1", "n2", "n3"]])
    tramos = np.array_split(np.unique(t), 6)
    modelos = CLASE + list(SOTA)
    hist_P, hist_y, hist_t, hist_G = [], [], [], []
    filas, preds = [], []
    for j in range(1, 6):
        tr = np.isin(t, np.concatenate(tramos[:j]))
        te = np.isin(t, tramos[j])
        print(conjunto, f"pliegue {j}: {tr.sum()} filas de entrenamiento, {te.sum()} de evaluación", flush=True)
        b = base_pliegue(D, tr, te, n_jobs)
        P_te = np.column_stack([b["te"][m] for m in modelos])
        # historial del combinador: tramos de evaluación anteriores o, en el pliegue 1, predicciones internas
        if j == 1:
            pos = b["oof_pos"]
            Ph, yh, th, Gh = np.column_stack([b["oof"][m] for m in modelos]), y[pos], t[pos], G[pos]
        else:
            Ph, yh, th, Gh = np.vstack(hist_P), np.concatenate(hist_y), np.concatenate(hist_t), np.vstack(hist_G)
        Xh = matriz(Ph, Gh)
        corte = np.quantile(th, 0.85)
        es_val = th > corte if (th > corte).any() and (th <= corte).any() else np.arange(len(th)) >= int(0.85 * len(th))
        obj = ObjetivoAJKL(Xh, yh, th, es_val)
        res = O.bayes_tpe(obj, ESPACIO_AJKL, C.PRESUPUESTO, C.SEMILLA)
        hist = pd.DataFrame(res["historial"])
        i = int(hist["puntaje"].to_numpy().argmax())
        mejor, calibrar = dict(hist.loc[i, "params"]), bool(hist.loc[i, "calibrar"])
        aj = ApilamientoJerarquicoKL(n_modelos=len(modelos), calibrar=calibrar, **mejor).fit(Xh, yh, tiempos=th)
        p_aj = aj.predict_proba(matriz(P_te, G[te]))[:, 1]
        glob = ApilamientoJerarquicoKL(n_modelos=len(modelos), calibrar=calibrar, **{**mejor, "profundidad": 1}).fit(Xh, yh, tiempos=th)
        p_glob = glob.predict_proba(matriz(P_te, G[te]))[:, 1]
        res_fila = {"conjunto": conjunto, "pliegue": j, "n_eval": int(te.sum()), "tasa": float(y[te].mean()),
                    "params_ajkl": {**mejor, "calibrar": calibrar}}
        for nombre, p in [("AJ-KL", p_aj), ("AJ-KL global (profundidad 1)", p_glob)] + [(m, b["te"][m]) for m in modelos]:
            res_fila[nombre] = roc_auc_score(y[te], p)
            res_fila[nombre + " | log_loss"] = log_loss(y[te], np.clip(p, 1e-7, 1 - 1e-7))
            res_fila[nombre + " | brier"] = float(np.mean((p - y[te]) ** 2))
        filas.append(res_fila)
        preds.append(pd.DataFrame({"pliegue": j, "tiempo": t[te], "y": y[te], "AJ-KL": p_aj, "AJ-KL global": p_glob,
                                   **{m: b["te"][m] for m in modelos}}))
        print(conjunto, j, {k: round(v, 4) for k, v in res_fila.items() if isinstance(v, float) and "|" not in k}, flush=True)
        hist_P.append(P_te); hist_y.append(y[te]); hist_t.append(t[te]); hist_G.append(G[te])
    salida = {"tabla": pd.DataFrame(filas), "pred": pd.concat(preds, ignore_index=True), "nombre": D["nombre"]}
    SAL.mkdir(parents=True, exist_ok=True)
    tmp = archivo.with_suffix(".tmp")
    joblib.dump(salida, tmp, compress=3)
    os.replace(tmp, archivo)
    return salida


if __name__ == "__main__":
    ejecutar(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
