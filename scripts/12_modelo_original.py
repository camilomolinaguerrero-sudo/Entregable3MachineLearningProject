"""Modelo original (EJUL) dentro de la misma validación cruzada anidada del Entregable 2.

Para cada pliegue externo (k1-k4 y final):
1. Bucle interno: los mismos tres pliegues temporales de ventana creciente dentro de S_k. El
   preprocesamiento se ajusta en cada entrenamiento interno.
2. El clasificador base de cada umbral es el XGBoost de referencia con los hiperparámetros que el
   bucle interno del Entregable 2 eligió para ese pliegue (XGBoost | Sin balanceo | Random Search).
   La codificación jerárquica es temporal (cada fila de entrenamiento se codifica solo con semanas
   anteriores), versión 2 del diseño; ver la sección 5.8 del libro sobre las iteraciones.
   TPE (Optuna, semilla 42, 24 evaluaciones, 8 de arranque) ajusta solo los hiperparámetros propios
   del EJUL con el mismo criterio que los benchmarks: el AUC medio de los pliegues internos. Como el
   calibrador es una transformación monótona y no cambia el AUC, se elige después, para la mejor
   configuración, entre Platt y beta según la log-loss interna (ambos se calculan en cada evaluación
   con los mismos miembros). Con `log_loss` como segundo argumento se repite el experimento usando
   la log-loss interna como criterio (análisis de sensibilidad).
3. Con la mejor configuración se elige el umbral de decisión (máximo F1 sobre las predicciones
   internas fuera de pliegue, igual que en el Entregable 2), se reajusta con todo S_k y se evalúa
   una sola vez en V_k.

Uso: python scripts/12_modelo_original.py k1 [auc|log_loss] [n_jobs]
Salida: data/resultados/original/<pliegue>.joblib (criterio AUC) u original_logloss/ (resumen,
historial y predicciones en V).
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
from ejul import EnsambleUmbralesLatentes  # noqa: E402

COMBO_BASE = "XGBoost | Sin balanceo | Random Search"
NOMBRE = "EJUL (original)"
ESPACIO = [
    M.Param("h", "float", 0.05, 0.40),
    M.Param("n_umbrales", "cat", opciones=(3, 5, 7, 9)),
    M.Param("nucleo", "cat", opciones=("uniforme", "epanechnikov", "triangular")),
    M.Param("grado", "cat", opciones=(0, 2)),
    M.Param("fuerza_eb", "float", 0.25, 4.0, log=True),
    M.Param("jerarquico", "cat", opciones=(True, False)),
]
CALIBRADORES = ("platt", "beta")
SALIDAS = {"auc": RAIZ / "data" / "resultados" / "original",
           "log_loss": RAIZ / "data" / "resultados" / "original_logloss"}
CODIFICACION = "temporal"


def codigos_grupo(panel: pd.DataFrame) -> pd.DataFrame:
    """Códigos enteros de corredor ⊂ municipio de origen ⊂ departamento (solo etiquetas)."""
    return pd.DataFrame({
        "g_corredor": pd.factorize(panel["mercado"].astype(str) + "|" + panel["cod_mpio"].astype(str))[0],
        "g_mpio": pd.factorize(panel["cod_mpio"].astype(str))[0],
        "g_depto": pd.factorize(panel["depto_origen"].astype(str))[0],
    }, index=panel["fila"].to_numpy())


def matriz(prep, df, G, ajustar=False):
    y = df[C.OBJ_CLF].to_numpy()
    Xm = prep.fit_transform(df, y) if ajustar else prep.transform(df)
    return np.c_[Xm, G.loc[df["fila"].to_numpy()].to_numpy()]


class ObjetivoEJUL:
    """Evalúa una configuración del EJUL en los pliegues internos (con caché de repetidas).

    Con los mismos miembros calcula las dos recalibraciones (Platt y beta); el AUC no depende de
    ellas. `criterio` = "auc" maximiza el AUC medio; "log_loss" minimiza la mejor log-loss media.
    """

    def __init__(self, internos, base, columnas_grupo, criterio="auc"):
        self.internos = internos
        self.base = base
        self.cg = columnas_grupo
        self.criterio = criterio
        self.cache, self.predicciones = {}, {}

    def __call__(self, params):
        from scipy.special import expit
        from ejul import logistica_newton, rasgos_calibracion
        clave = O._clave(params)
        if clave in self.cache:
            return {**self.cache[clave], "repetida": True, "segundos": 0.0}
        t0 = time.perf_counter()
        ll = {c: [] for c in CALIBRADORES}
        auc, preds = [], {c: [] for c in CALIBRADORES}
        for X_tr, y_tr, r_tr, t_tr, X_va, y_va in self.internos:
            m = EnsambleUmbralesLatentes(estimador_base=self.base, columnas_grupo=self.cg, random_state=C.SEMILLA,
                                         calibrador="platt", codificacion=CODIFICACION, **params)
            m.fit(X_tr, y_tr, latente=r_tr, tiempos=t_tr)
            crudo = m.probabilidad_cruda(X_va)
            ok = np.isfinite(m.oof_crudo_)
            for c in CALIBRADORES:
                w = logistica_newton(rasgos_calibracion(m.oof_crudo_[ok], c), y_tr[ok])
                p = expit(w[0] + rasgos_calibracion(crudo, c) @ w[1:])
                ll[c].append(log_loss(y_va, p))
                preds[c].append(p.astype(np.float32))
            auc.append(roc_auc_score(y_va, preds["platt"][-1]))
        ll_media = {c: float(np.mean(v)) for c, v in ll.items()}
        cal = min(ll_media, key=ll_media.get)
        puntaje = float(np.mean(auc)) if self.criterio == "auc" else -ll_media[cal]
        r = {"puntaje": puntaje, "auc_interno": float(np.mean(auc)), "log_loss_interno": ll_media[cal],
             "log_loss_platt": ll_media["platt"], "log_loss_beta": ll_media["beta"], "calibrador": cal,
             "segundos": time.perf_counter() - t0, "repetida": False}
        self.cache[clave] = {k: v for k, v in r.items() if k not in ("segundos", "repetida")}
        self.predicciones[clave] = preds
        return r


def ejecutar(pliegue: str, criterio: str = "auc", n_jobs: int = 2) -> dict:
    SALIDA = SALIDAS[criterio]
    archivo = SALIDA / f"{pliegue}.joblib"
    if archivo.exists():
        return joblib.load(archivo)
    warnings.filterwarnings("ignore")
    C.fijar_semillas(C.SEMILLA)
    t_ini = time.perf_counter()
    panel = D.cargar_panel()
    G = codigos_grupo(panel)
    razon = pd.Series(panel["ratio_t1"].to_numpy(), index=panel["fila"].to_numpy())
    maestra = pd.read_parquet(RAIZ / "data" / "resultados" / "maestra.parquet")
    fila_base = maestra[(maestra.tarea == "clf") & (maestra.combinacion == COMBO_BASE) & (maestra.pliegue == pliegue)].iloc[0]
    params_base = json.loads(fila_base["params_json"])
    base = M.construir(M.catalogo("clf")["XGBoost"], params_base)
    base.set_params(n_jobs=n_jobs)

    S, V = X.cargar_pliegue(pliegue)
    # Bucle interno (mismos pliegues temporales que el Entregable 2)
    t0 = time.perf_counter()
    internos, y_int = [], []
    for tr, va in D.pliegues_internos(S["semana_t1"]):
        Str, Sva = S.iloc[tr], S.iloc[va]
        prep = P.preprocesador("arboles", "clf")
        internos.append((matriz(prep, Str, G, True), Str[C.OBJ_CLF].to_numpy(), razon.loc[Str["fila"]].to_numpy(),
                         Str["semana_t1"].to_numpy(), matriz(prep, Sva, G), Sva[C.OBJ_CLF].to_numpy()))
        y_int.append(Sva[C.OBJ_CLF].to_numpy())
    cg = (-3, -2, -1)        # las tres últimas columnas: corredor, municipio y departamento
    t_cache = time.perf_counter() - t0
    obj = ObjetivoEJUL(internos, base, cg, criterio)
    t0 = time.perf_counter()
    res = O.bayes_tpe(obj, ESPACIO, C.PRESUPUESTO, C.SEMILLA)
    t_busqueda = time.perf_counter() - t0
    hist = pd.DataFrame(res["historial"])
    i_mejor = int(hist["puntaje"].to_numpy().argmax())
    calibrador = hist.loc[i_mejor, "calibrador"]
    mejor = {**hist.loc[i_mejor, "params"], "calibrador": calibrador}
    s_int = np.concatenate(obj.predicciones[O._clave(hist.loc[i_mejor, "params"])][calibrador]).astype(float)
    umbral = EV.umbral_f1(np.concatenate(y_int), s_int)

    # Reajuste con todo S y evaluación única en V
    t0 = time.perf_counter()
    prep = P.preprocesador("arboles", "clf")
    X_S = matriz(prep, S, G, True)
    y_S = S[C.OBJ_CLF].to_numpy()
    modelo = EnsambleUmbralesLatentes(estimador_base=base, columnas_grupo=cg, random_state=C.SEMILLA,
                                      codificacion=CODIFICACION, **mejor)
    modelo.fit(X_S, y_S, latente=razon.loc[S["fila"]].to_numpy(), tiempos=S["semana_t1"].to_numpy())
    t_refit = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_V = modelo.predict_proba(matriz(prep, V, G))[:, 1]
    t_pred = time.perf_counter() - t0
    y_V = V[C.OBJ_CLF].to_numpy()
    resumen = {"tarea": "clf", "modelo": NOMBRE, "balanceo": "Sin balanceo", "optimizador": "Bayesiana (TPE)",
               "pliegue": pliegue, "semilla": C.SEMILLA, "criterio": criterio, "codificacion": CODIFICACION, "presupuesto": C.PRESUPUESTO, "evaluaciones": len(hist),
               "params": mejor, "params_base": params_base, "puntaje_interno": float(hist.loc[i_mejor, "puntaje"]),
               "auc_interno": float(hist.loc[i_mejor, "auc_interno"]),
               "t_cache": t_cache, "t_busqueda": t_busqueda, "t_por_evaluacion": float(hist["segundos"].mean()),
               "t_refit": t_refit, "t_pred": t_pred, "t_total": time.perf_counter() - t_ini,
               "filas_entrenamiento": int(len(y_S)), "filas_eval": int(len(V)), "n_variables": int(X_S.shape[1]),
               "pesos": modelo.pesos_.tolist(), "umbrales_latentes": modelo.umbrales_.tolist(),
               "calibracion": None if modelo.w_cal_ is None else modelo.w_cal_.tolist()}
    resumen.update(EV.metricas_clf(y_V, p_V, umbral, p_V))
    salida = {"resumen": resumen, "historial": hist, "pred": {"fila": V["fila"].to_numpy(), "s": p_V.astype(np.float32)}}
    SALIDA.mkdir(parents=True, exist_ok=True)
    tmp = archivo.with_suffix(".tmp")
    joblib.dump(salida, tmp, compress=3)
    os.replace(tmp, archivo)
    return salida


if __name__ == "__main__":
    pl = sys.argv[1]
    crit = sys.argv[2] if len(sys.argv) > 2 else "auc"
    nj = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    out = ejecutar(pl, crit, nj)
    r = out["resumen"]
    print(pl, {k: round(r[k], 4) for k in ["auc", "auc_pr", "brier", "log_loss", "ece", "f1", "precision", "recall"]},
          r["params"], f"{r['t_total']:.0f}s", flush=True)
