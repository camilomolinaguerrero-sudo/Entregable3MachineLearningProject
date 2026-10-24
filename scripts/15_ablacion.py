"""Estudio de ablación del EJUL y contraste de su regla de combinación con reglas estándar.

En cada pliegue externo se parte de la configuración elegida por TPE (script 12) y se retira un
componente a la vez. Los miembros, la recalibración y los datos son los mismos en todas las
variantes que solo cambian la regla de combinación (se calculan con un único ajuste).

Variantes con el mismo ajuste:
  completo                  la configuración elegida
  sin_recalibracion         probabilidad combinada antes del calibrador
  sin_reordenamiento        sin la proyección isotónica (calibrador reajustado)
  pesos_grado_alterno       grado 0 ↔ 2 del polinomio local (calibrador reajustado)
  regla_promedio_simple     ω_k = 1/K
  regla_error_global        ω_k ∝ 1 / Brier_k fuera de pliegue (respecto de la etiqueta en τ)
  regla_stacking_lineal     regresión logística sobre logit(g_k) ajustada con las predicciones
                            fuera de pliegue (meta-aprendiz estándar)
Variantes con un ajuste propio:
  sin_umbrales              h = 0 (un solo miembro en τ): codificación jerárquica + recalibración
  sin_jerarquico            sin la codificación Bayes empírica de corredor, municipio y departamento
  sin_diversificar          todos los miembros con la misma semilla
  codificacion_cruzada      codificación jerárquica con ajuste cruzado aleatorio (versión 1 del diseño)
  control_semillas_xgb      K XGBoost en τ con semillas distintas (reducción de varianza pura)
  control_regresion         XGBoost de regresión de log(1 + R) + distribución empírica de residuos
                            fuera de pliegue: P(R < τ | x) = F_e(log(1 + τ) − m̂(x))
Uso: python scripts/15_ablacion.py k1 [n_jobs]
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import KFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))
import config as C  # noqa: E402
import datos as D  # noqa: E402
import evaluacion as EV  # noqa: E402
import experimento as X  # noqa: E402
import modelos as M  # noqa: E402
import preprocesamiento as P  # noqa: E402
from ejul import (EnsambleUmbralesLatentes, logistica_newton, nucleo,  # noqa: E402
                            pesos_polinomio_local, proyeccion_isotonica, rasgos_calibracion)

RES = RAIZ / "data" / "resultados"
SAL = RES / "ablacion"
METR = ["auc", "auc_pr", "brier", "log_loss", "ece", "f1", "precision", "recall", "accuracy"]


def importar_12():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m12", RAIZ / "scripts" / "12_modelo_original.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def calibrar(oof, y, p, tipo):
    ok = np.isfinite(oof)
    w = logistica_newton(rasgos_calibracion(oof[ok], tipo), y[ok])
    f = lambda q: expit(w[0] + rasgos_calibracion(q, tipo) @ w[1:])
    return f(p), np.where(ok, f(np.where(ok, oof, 0.5)), np.nan)


def metricas(y_V, p_V, y_S, oof_cal):
    ok = np.isfinite(oof_cal)
    umbral = EV.umbral_f1(y_S[ok], oof_cal[ok])
    r = EV.metricas_clf(y_V, p_V, umbral, p_V)
    return {k: r[k] for k in METR}


def ejecutar(pliegue, n_jobs=4):
    warnings.filterwarnings("ignore")
    C.fijar_semillas(C.SEMILLA)
    m12 = importar_12()
    res = joblib.load(RES / "original" / f"{pliegue}.joblib")["resumen"]
    cfg = dict(res["params"])
    panel = D.cargar_panel()
    G = m12.codigos_grupo(panel)
    razon = pd.Series(panel["ratio_t1"].to_numpy(), index=panel["fila"].to_numpy())
    yreg = pd.Series(panel[C.OBJ_REG].to_numpy(), index=panel["fila"].to_numpy())
    base = M.construir(M.catalogo("clf")["XGBoost"], res["params_base"])
    base.set_params(n_jobs=n_jobs)
    S, V = X.cargar_pliegue(pliegue)
    prep = P.preprocesador("arboles", "clf")
    X_S, X_V = m12.matriz(prep, S, G, True), m12.matriz(prep, V, G)
    y_S, y_V = S[C.OBJ_CLF].to_numpy(), V[C.OBJ_CLF].to_numpy()
    R_S = razon.loc[S["fila"]].to_numpy()
    T_S = S["semana_t1"].to_numpy()
    cg = (-3, -2, -1)
    cod = res.get("codificacion", "cruzada")
    filas, preds = [], {}

    def registrar(nombre, p_V, oof_cal, t):
        filas.append({"pliegue": pliegue, "variante": nombre, **metricas(y_V, p_V, y_S, oof_cal), "segundos": t})
        preds[nombre] = p_V.astype(np.float32)
        print(pliegue, nombre, {k: round(v, 4) for k, v in filas[-1].items() if k in METR[:5]}, flush=True)

    # ------------------------------------------------------------ ajuste completo
    t0 = time.perf_counter()
    mod = EnsambleUmbralesLatentes(estimador_base=base, columnas_grupo=cg, random_state=C.SEMILLA, codificacion=cod, **cfg)
    mod.fit(X_S, y_S, latente=R_S, tiempos=T_S)
    t_comp = time.perf_counter() - t0
    registrar("completo", mod.predict_proba(X_V)[:, 1], mod.oof_, t_comp)
    Pm_V = mod.probabilidades_miembros(X_V)
    Pm_S = mod.oof_miembros_
    ok = np.isfinite(Pm_S[:, 0])
    tipo = cfg["calibrador"]
    reord = lambda P: proyeccion_isotonica(P) if P.shape[1] > 1 else P
    comb = lambda P, w: np.clip(P @ w, 1e-6, 1 - 1e-6)

    registrar("sin_recalibracion", mod.probabilidad_cruda(X_V), np.where(ok, comb(reord(np.nan_to_num(Pm_S, nan=0.5)), mod.pesos_), np.nan), 0.0)
    if Pm_V.shape[1] > 1:
        oof_raw = np.where(ok, comb(np.nan_to_num(Pm_S, nan=0.5), mod.pesos_), np.nan)
        p, o = calibrar(oof_raw, y_S, comb(Pm_V, mod.pesos_), tipo)
        registrar("sin_reordenamiento", p, o, 0.0)
        u = mod.u_
        alterno = 2 if cfg["grado"] == 0 else 0
        w_alt = pesos_polinomio_local(u, nucleo(u, cfg["nucleo"]), alterno)
        oof_alt = np.where(ok, comb(reord(np.nan_to_num(Pm_S, nan=0.5)), w_alt), np.nan)
        p, o = calibrar(oof_alt, y_S, comb(reord(Pm_V), w_alt), tipo)
        registrar(f"pesos_grado_{alterno}", p, o, 0.0)
        K = Pm_V.shape[1]
        w_s = np.full(K, 1 / K)
        p, o = calibrar(np.where(ok, comb(np.nan_to_num(Pm_S, nan=0.5), w_s), np.nan), y_S, comb(Pm_V, w_s), tipo)
        registrar("regla_promedio_simple", p, o, 0.0)
        brier_k = np.mean((Pm_S[ok] - y_S[ok, None]) ** 2, axis=0)
        w_e = (1 / brier_k) / np.sum(1 / brier_k)
        p, o = calibrar(np.where(ok, comb(np.nan_to_num(Pm_S, nan=0.5), w_e), np.nan), y_S, comb(Pm_V, w_e), tipo)
        registrar("regla_error_global", p, o, 0.0)
        Lg = lambda P: logit(np.clip(P, 1e-6, 1 - 1e-6))
        w_st = logistica_newton(Lg(Pm_S[ok]), y_S[ok], ridge=1e-3)
        p_st = expit(w_st[0] + Lg(Pm_V) @ w_st[1:])
        o_st = np.full(len(y_S), np.nan)
        o_st[ok] = expit(w_st[0] + Lg(Pm_S[ok]) @ w_st[1:])     # (dentro de muestra del meta-aprendiz)
        registrar("regla_stacking_lineal", p_st, o_st, 0.0)

    # ------------------------------------------------------------ variantes con ajuste propio
    variantes = [("sin_umbrales", {"h": 0.0}), ("sin_jerarquico", {"jerarquico": False}),
                 ("sin_diversificar", {"diversificar": False}), ("prediccion_por_pliegues", {"prediccion": "pliegues"})]
    if cod == "temporal":
        variantes.append(("codificacion_cruzada", {"codificacion": "cruzada"}))
    for nombre, cambio in variantes:
        t0 = time.perf_counter()
        m = EnsambleUmbralesLatentes(estimador_base=base, columnas_grupo=cg, random_state=C.SEMILLA,
                                     **{**cfg, "codificacion": cod, **cambio})
        m.fit(X_S, y_S, latente=R_S, tiempos=T_S)
        registrar(nombre, m.predict_proba(X_V)[:, 1], m.oof_, time.perf_counter() - t0)

    # control: K XGBoost en τ con semillas distintas (mismo número de miembros, sin umbrales ni grupos)
    t0 = time.perf_counter()
    K = len(mod.umbrales_)
    Xb_S, Xb_V = X_S[:, :-3], X_V[:, :-3]
    ps = []
    for k in range(K):
        mk = M.construir(M.catalogo("clf")["XGBoost"], res["params_base"], semilla=C.SEMILLA + k)
        mk.set_params(n_jobs=n_jobs)
        ps.append(mk.fit(Xb_S, y_S).predict_proba(Xb_V)[:, 1])
    oof_x = np.full(len(y_S), np.nan)
    for tr, va in KFold(3, shuffle=True, random_state=C.SEMILLA).split(Xb_S):
        mk = M.construir(M.catalogo("clf")["XGBoost"], res["params_base"])
        mk.set_params(n_jobs=n_jobs)
        oof_x[va] = mk.fit(Xb_S[tr], y_S[tr]).predict_proba(Xb_S[va])[:, 1]
    registrar("control_semillas_xgb", np.mean(ps, axis=0), oof_x, time.perf_counter() - t0)

    # control: regresión de log(1 + R) y distribución empírica de residuos fuera de pliegue
    t0 = time.perf_counter()
    maestra = pd.read_parquet(RES / "maestra.parquet")
    reg = maestra[(maestra.tarea == "reg") & (maestra.modelo == "XGBoost") & (maestra.pliegue == pliegue)]
    reg = reg.sort_values("rmse").iloc[0]
    preg = P.preprocesador("arboles", "reg")
    yr_S = yreg.loc[S["fila"]].to_numpy()
    Xr_S, Xr_V = preg.fit_transform(S, yr_S), preg.transform(V)
    espec_r = M.catalogo("reg")["XGBoost"]
    mr = M.construir(espec_r, json.loads(reg["params_json"]))
    mr.set_params(n_jobs=n_jobs)
    mr.fit(Xr_S, yr_S)
    pred_oof = np.empty(len(yr_S))
    for tr, va in KFold(3, shuffle=True, random_state=C.SEMILLA).split(Xr_S):
        mk = M.construir(espec_r, json.loads(reg["params_json"]))
        mk.set_params(n_jobs=n_jobs)
        pred_oof[va] = mk.fit(Xr_S[tr], yr_S[tr]).predict(Xr_S[va])
    resid = np.sort(yr_S - pred_oof)                       # residuos fuera de pliegue
    corte = np.log1p(C.UMBRAL_INTERRUPCION)
    F = lambda q: np.clip(np.searchsorted(resid, q, side="left") / len(resid), 1e-4, 1 - 1e-4)
    registrar("control_regresion", F(corte - mr.predict(Xr_V)), F(corte - pred_oof), time.perf_counter() - t0)

    SAL.mkdir(parents=True, exist_ok=True)
    out = {"tabla": pd.DataFrame(filas), "pred": {"fila": V["fila"].to_numpy(), **preds}}
    joblib.dump(out, SAL / f"{pliegue}.joblib", compress=3)
    return out


if __name__ == "__main__":
    pl = sys.argv[1]
    nj = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ejecutar(pl, nj)
