"""Modelo original AJ-KL (versión 2) dentro de la misma validación cruzada anidada del Entregable 2.

Modelos base (M = 9): las mejores combinaciones de los 7 modelos del Entregable 2 y los dos comparadores
recientes del estado del arte (LightGBM y CatBoost), todos ajustados con el mismo protocolo (mismos
pliegues, misma muestra S_k, TPE con 24 evaluaciones y semilla 42). Sus probabilidades en cada bloque
externo V_k ya existen: son las de los modelos reentrenados con todo S_k.

Historial de predicciones fuera de muestra con que se entrena el combinador del pliegue k:
  * k = k2, k3, k4, final: las predicciones de los bloques externos anteriores (V_1, ..., V_{k−1}), que
    salen de modelos entrenados con 40 mil filas, el mismo régimen que las predicciones de V_k que se van a
    combinar. Todas sus semanas son anteriores a V_k (ninguna fuga).
  * k = k1 (sin bloque anterior): las predicciones de las tres ventanas internas temporales de S_k1.
Jerarquía: país → central → central × grupo de alimentos → corredor (central × grupo × municipio).

Ajuste de los hiperparámetros propios: TPE (Optuna, semilla 42, 24 evaluaciones, 8 de arranque), con el
AUC como criterio, como los benchmarks. Validación hacia adelante dentro del historial: se entrena con las
semanas anteriores y se evalúa en las últimas 26 semanas del historial (en k1, entre ventanas internas).
La calibración se decide por log-loss en esa misma validación.
Uso: python scripts/22_ajkl.py <pliegue> [n_jobs]
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import logit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config as C  # noqa: E402
import datos as D  # noqa: E402
import evaluacion as EV  # noqa: E402
import experimento as X  # noqa: E402
import modelos as M  # noqa: E402
import optimizacion as O  # noqa: E402
import preprocesamiento as P  # noqa: E402
from original_model import ApilamientoJerarquicoKL  # noqa: E402

RES = RAIZ / "data" / "resultados"
SAL = RES / "ajkl"
NOMBRE = "AJ-KL (original)"
ORDEN = ["k1", "k2", "k3", "k4", "final"]
SOTA = ["LightGBM", "CatBoost"]
ESPACIO = [
    M.Param("lam", "float", 0.01, 1000.0, log=True),
    M.Param("gamma", "float", 0.98, 1.0),
    M.Param("profundidad", "cat", opciones=(1, 2, 3, 4)),
]
# η0 y el número de iteraciones ya no son hiperparámetros: cada nodo se resuelve hasta la brecha de
# Frank-Wolfe (versión 2.1; en la 2.0 la parada en K ≤ 300 dejaba los pesos lejos del óptimo).
SEMANAS_VALIDACION = 26


def seleccion():
    return pd.read_csv(RES / "seleccion_mejores.csv")


def columnas_base(sel):
    return sel["combinacion"].tolist() + SOTA


def codigos_jerarquia(panel):
    """Códigos enteros (solo etiquetas) de los tres niveles bajo la raíz, del más grueso al más fino."""
    c = panel["mercado"].astype(str)
    cg = c + "|" + panel["grupo_dom"].astype(str)
    return pd.DataFrame({"g_central": pd.factorize(c)[0], "g_central_grupo": pd.factorize(cg)[0],
                         "g_corredor": pd.factorize(cg + "|" + panel["cod_mpio"].astype(str))[0]},
                        index=panel["fila"].to_numpy())


def prob_bloque(pliegue, sel):
    """Probabilidades de los 9 modelos base en V_k, alineadas con las filas de V_k."""
    _, V = X.cargar_pliegue(pliegue)
    archivo = "pred_prueba_clf.parquet" if pliegue == "final" else "pred_externo_clf.parquet"
    pred = pd.read_parquet(RES / archivo)
    if pliegue != "final":
        pred = pred[pred["pliegue"] == pliegue]
    pred = pred.set_index("fila").loc[V["fila"]]
    out = pd.DataFrame({"fila": V["fila"].to_numpy(), "semana_t1": V["semana_t1"].to_numpy(),
                        C.OBJ_CLF: V[C.OBJ_CLF].to_numpy()})
    for c in sel["combinacion"]:
        out[c] = (pred[c + " | prob"] if (c + " | prob") in pred.columns else pred[c]).to_numpy()
    for m in SOTA:
        p = joblib.load(RES / "sota" / f"{m}_{pliegue}.joblib")["pred"]
        out[m] = pd.Series(p["s"], index=p["fila"]).loc[V["fila"]].to_numpy()
    return out


def construir_sota(nombre, params, n_jobs=2):
    if nombre == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, subsample_freq=1, random_state=C.SEMILLA, n_jobs=n_jobs, verbose=-1, **params)
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=300, random_seed=C.SEMILLA, thread_count=n_jobs, verbose=0,
                              allow_writing_files=False, **params)


def oof_internas(pliegue, sel, n_jobs=2):
    """Predicciones de los 9 modelos en las tres ventanas internas temporales de S_k (solo para k1)."""
    ruta = SAL / f"oof_internas_{pliegue}.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)
    S, _ = X.cargar_pliegue(pliegue)
    maestra = pd.read_parquet(RES / "maestra.parquet")
    pl = list(D.pliegues_internos(S["semana_t1"]))
    out = pd.DataFrame({"fila": np.concatenate([S["fila"].to_numpy()[va] for _, va in pl]),
                        "ventana": np.concatenate([np.full(len(va), j) for j, (_, va) in enumerate(pl)])})
    out["semana_t1"] = np.concatenate([S["semana_t1"].to_numpy()[va] for _, va in pl])
    out[C.OBJ_CLF] = np.concatenate([S[C.OBJ_CLF].to_numpy()[va] for _, va in pl])
    y = out[C.OBJ_CLF].to_numpy()
    for r in sel.itertuples():
        espec = M.catalogo("clf")[r.modelo]
        params = json.loads(maestra[(maestra.tarea == "clf") & (maestra.combinacion == r.combinacion) &
                                    (maestra.pliegue == pliegue)].iloc[0]["params_json"])
        pliegues, _ = X.cache_interna(espec, S, r.balanceo, C.SEMILLA)
        s = np.concatenate([M.puntaje(M.construir(espec, params, r.balanceo, y_tr, C.SEMILLA).fit(X_tr, y_tr), X_va)
                            for X_tr, y_tr, X_va, _ in pliegues]).astype(float)
        out[r.combinacion] = s if M.es_probabilistico(espec) else EV.Platt().fit(s, y).predict(s)
    for m in SOTA:
        params = joblib.load(RES / "sota" / f"{m}_{pliegue}.joblib")["resumen"]["params"]
        s = []
        for tr, va in pl:
            prep = P.preprocesador("arboles", "clf")
            Xtr = prep.fit_transform(S.iloc[tr], S[C.OBJ_CLF].to_numpy()[tr])
            s.append(construir_sota(m, params, n_jobs).fit(Xtr, S[C.OBJ_CLF].to_numpy()[tr]).predict_proba(prep.transform(S.iloc[va]))[:, 1])
        out[m] = np.concatenate(s)
    SAL.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ruta, index=False)
    return out


def historial(pliegue, sel, n_jobs=2):
    """Historial de predicciones fuera de muestra anteriores a V_k y su partición de validación."""
    j = ORDEN.index(pliegue)
    if j == 0:
        h = oof_internas(pliegue, sel, n_jobs)
        return h, [(h["ventana"] < v, h["ventana"] == v) for v in (1, 2)]
    h = pd.concat([prob_bloque(o, sel).assign(bloque=o) for o in ORDEN[:j]], ignore_index=True)
    S, _ = X.cargar_pliegue(pliegue)
    h = h[pd.to_datetime(h["semana_t1"]) <= S["semana_t1"].max()].reset_index(drop=True)   # respeta el hueco
    corte = pd.to_datetime(h["semana_t1"]).max() - pd.Timedelta(weeks=SEMANAS_VALIDACION)
    va = pd.to_datetime(h["semana_t1"]) > corte
    return h, [(~va, va)]


def matriz(df, cols, G):
    return np.c_[np.clip(df[cols].to_numpy(float), 1e-4, 1 - 1e-4), G.loc[df["fila"]].to_numpy()]


def semanas(s):
    return ((pd.to_datetime(s) - pd.Timestamp("2018-01-01")).dt.days // 7).to_numpy(float)


class ObjetivoAJKL:
    def __init__(self, h, particiones, cols, G):
        self.h, self.part, self.cols, self.G = h, particiones, cols, G
        self.X = matriz(h, cols, G)
        self.y = h[C.OBJ_CLF].to_numpy()
        self.t = semanas(h["semana_t1"])
        self.cache, self.predicciones = {}, {}

    def __call__(self, params):
        clave = O._clave(params)
        if clave in self.cache:
            return {**self.cache[clave], "repetida": True, "segundos": 0.0}
        t0 = time.perf_counter()
        aucs, ll_cal, ll_raw, preds = [], [], [], []
        for tr, va in self.part:
            tr, va = np.asarray(tr), np.asarray(va)
            m = ApilamientoJerarquicoKL(n_modelos=len(self.cols), calibrar=True, **params)
            m.fit(self.X[tr], self.y[tr], tiempos=self.t[tr])
            p = m.predict_proba(self.X[va])[:, 1]
            q = m.probabilidad_pool(self.X[va])
            aucs.append(roc_auc_score(self.y[va], q))
            ll_cal.append(log_loss(self.y[va], p))
            ll_raw.append(log_loss(self.y[va], q))
            preds.append({"cal": p, "raw": q, "y": self.y[va]})
        calibrar = bool(np.mean(ll_cal) <= np.mean(ll_raw))
        r = {"puntaje": float(np.mean(aucs)), "auc_interno": float(np.mean(aucs)), "calibrar": calibrar,
             "log_loss_interno": float(min(np.mean(ll_cal), np.mean(ll_raw))), "segundos": time.perf_counter() - t0,
             "repetida": False}
        self.cache[clave] = {k: v for k, v in r.items() if k not in ("segundos", "repetida")}
        self.predicciones[clave] = (np.concatenate([d["cal" if calibrar else "raw"] for d in preds]),
                                    np.concatenate([d["y"] for d in preds]))
        return r


def blp(P_, y, P_nuevo):
    """Pool lineal transformado con beta (Ranjan y Gneiting, 2010) por máxima verosimilitud."""
    M_ = P_.shape[1]

    def nll(theta):
        w = softmax(theta[:M_])
        q = np.clip(stats.beta.cdf(np.clip(P_ @ w, 1e-6, 1 - 1e-6), np.exp(theta[M_]), np.exp(theta[M_ + 1])), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log1p(-q))

    th = optimize.minimize(nll, np.zeros(M_ + 2), method="L-BFGS-B").x
    w = softmax(th[:M_])
    f = lambda Q: stats.beta.cdf(np.clip(Q @ w, 1e-6, 1 - 1e-6), np.exp(th[M_]), np.exp(th[M_ + 1]))
    return f(P_nuevo), f(P_)


def reglas(h, cols, V):
    """Reglas estándar ajustadas con el mismo historial: (predicción en V, predicción en el historial)."""
    Ph = np.clip(h[cols].to_numpy(float), 1e-4, 1 - 1e-4)
    Pv = np.clip(V[cols].to_numpy(float), 1e-4, 1 - 1e-4)
    y = h[C.OBJ_CLF].to_numpy()
    out = {"Promedio simple": (Pv.mean(axis=1), Ph.mean(axis=1))}
    ll = np.array([log_loss(y, Ph[:, m]) for m in range(Ph.shape[1])])
    w = (1 / ll) / np.sum(1 / ll)
    out["Pesos por inverso de la log-loss"] = (Pv @ w, Ph @ w)
    st = LogisticRegression(C=1.0, max_iter=3000).fit(logit(Ph), y)
    out["Stacking logístico"] = (st.predict_proba(logit(Pv))[:, 1], st.predict_proba(logit(Ph))[:, 1])
    out["BLP (Ranjan y Gneiting)"] = blp(Ph, y, Pv)
    return out


def metricas(y, p, umbral):
    r = EV.metricas_clf(y, p, umbral, np.clip(p, 1e-7, 1 - 1e-7))
    return {k: r[k] for k in ["auc", "auc_pr", "brier", "log_loss", "ece", "f1", "precision", "recall", "accuracy", "umbral"]}


def ejecutar(pliegue, n_jobs=2):
    archivo = SAL / f"{pliegue}.joblib"
    if archivo.exists():
        return joblib.load(archivo)
    warnings.filterwarnings("ignore")
    C.fijar_semillas(C.SEMILLA)
    t_ini = time.perf_counter()
    sel = seleccion()
    cols = columnas_base(sel)
    G = codigos_jerarquia(D.cargar_panel())
    h, part = historial(pliegue, sel, n_jobs)
    V = prob_bloque(pliegue, sel)
    obj = ObjetivoAJKL(h, part, cols, G)
    t0 = time.perf_counter()
    res = O.bayes_tpe(obj, ESPACIO, C.PRESUPUESTO, C.SEMILLA)
    t_busqueda = time.perf_counter() - t0
    hist = pd.DataFrame(res["historial"])
    i = int(hist["puntaje"].to_numpy().argmax())
    mejor = dict(hist.loc[i, "params"])
    calibrar = bool(hist.loc[i, "calibrar"])
    s_int, y_int = obj.predicciones[O._clave(mejor)]
    umbral = EV.umbral_f1(y_int, s_int)
    Xh, Xv = matriz(h, cols, G), matriz(V, cols, G)
    yh, yv = h[C.OBJ_CLF].to_numpy(), V[C.OBJ_CLF].to_numpy()
    th = semanas(h["semana_t1"])
    t0 = time.perf_counter()
    modelo = ApilamientoJerarquicoKL(n_modelos=len(cols), calibrar=calibrar, **mejor).fit(Xh, yh, tiempos=th)
    t_refit = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_V = modelo.predict_proba(Xv)[:, 1]
    t_pred = time.perf_counter() - t0
    resumen = {"tarea": "clf", "modelo": NOMBRE, "balanceo": "Sin balanceo", "optimizador": "Bayesiana (TPE)",
               "pliegue": pliegue, "semilla": C.SEMILLA, "presupuesto": C.PRESUPUESTO, "evaluaciones": len(hist),
               "params": {**mejor, "calibrar": calibrar}, "puntaje_interno": float(hist.loc[i, "puntaje"]),
               "historial_bloques": ("ventanas internas de S_k1" if pliegue == "k1" else ", ".join(ORDEN[:ORDEN.index(pliegue)])),
               "filas_historial": int(len(h)), "t_busqueda": t_busqueda, "t_por_evaluacion": float(hist["segundos"].mean()),
               "t_refit": t_refit, "t_pred": t_pred, "t_total": time.perf_counter() - t_ini,
               "filas_eval": int(len(V)), "modelos_base": cols, "nodos": [len(d) for d in modelo.pesos_]}
    resumen.update(metricas(yv, p_V, umbral))
    filas = [{"variante": "AJ-KL (completo)", **metricas(yv, p_V, umbral)}]
    for nombre, (pv, ph) in reglas(h, cols, V).items():
        filas.append({"variante": nombre, **metricas(yv, pv, EV.umbral_f1(yh, ph))})
    for nombre, cambio in [("Profundidad 1: apilamiento global", {"profundidad": 1}),
                           ("Profundidad 2: país y central", {"profundidad": 2}),
                           ("Profundidad 3: sin nivel de corredor", {"profundidad": 3}),
                           ("Profundidad 4: con nivel de corredor", {"profundidad": 4}),
                           ("Sin encogimiento (λ = 0)", {"lam": 0.0}), ("Sin olvido (γ = 1)", {"gamma": 1.0}),
                           ("Sin calibración", {"calibrar": False})]:
        m = ApilamientoJerarquicoKL(n_modelos=len(cols), **{**mejor, "calibrar": calibrar, **cambio}).fit(Xh, yh, tiempos=th)
        filas.append({"variante": nombre, **metricas(yv, m.predict_proba(Xv)[:, 1], EV.umbral_f1(yh, m.predict_proba(Xh)[:, 1]))})
    # sin los comparadores recientes: solo los 7 modelos del curso como base
    c7 = sel["combinacion"].tolist()
    m = ApilamientoJerarquicoKL(n_modelos=len(c7), **{**mejor, "calibrar": calibrar}).fit(matriz(h, c7, G), yh, tiempos=th)
    filas.append({"variante": "Solo los 7 modelos del curso como base", **metricas(
        yv, m.predict_proba(matriz(V, c7, G))[:, 1], EV.umbral_f1(yh, m.predict_proba(matriz(h, c7, G))[:, 1]))})
    # con el mismo número de filas etiquetadas que los benchmarks (40 mil, muestra aleatoria del historial)
    if len(h) > 40_000:
        sub = np.sort(np.random.default_rng(C.SEMILLA).choice(len(h), 40_000, replace=False))
        m = ApilamientoJerarquicoKL(n_modelos=len(cols), **{**mejor, "calibrar": calibrar}).fit(Xh[sub], yh[sub], tiempos=th[sub])
        filas.append({"variante": "Historial reducido a 40 mil filas", **metricas(
            yv, m.predict_proba(Xv)[:, 1], EV.umbral_f1(yh[sub], m.predict_proba(Xh[sub])[:, 1]))})
    for c in cols:
        filas.append({"variante": f"Modelo base: {c.split(' | ')[0]}", **metricas(yv, V[c].to_numpy(float), EV.umbral_f1(yh, h[c].to_numpy(float)))})
    salida = {"resumen": resumen, "historial": hist, "pred": {"fila": V["fila"].to_numpy(), "s": p_V.astype(np.float32)},
              "comparacion": pd.DataFrame(filas), "modelo": modelo}
    SAL.mkdir(parents=True, exist_ok=True)
    tmp = archivo.with_suffix(".tmp")
    joblib.dump(salida, tmp, compress=3)
    os.replace(tmp, archivo)
    return salida


if __name__ == "__main__":
    pl = sys.argv[1]
    out = ejecutar(pl, int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    r = out["resumen"]
    print(pl, {k: round(r[k], 4) for k in ["auc", "auc_pr", "brier", "log_loss", "ece", "f1"]}, r["params"], f"{r['t_total']:.0f}s")
    print(out["comparacion"].set_index("variante")[["auc", "auc_pr", "brier", "log_loss", "ece"]].round(4).to_string())
