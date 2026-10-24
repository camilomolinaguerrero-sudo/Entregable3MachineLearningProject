"""Robustez del EJUL frente a XGBoost (guía, cuadro 1: sensibilidad, semillas, ruido, desbalance,
tamaño de muestra). Todo se evalúa en los mismos pliegues externos, con los hiperparámetros elegidos
en el bucle interno de cada pliegue (script 12) y el XGBoost de referencia con los suyos.

Experimentos (uno por invocación):
  sensibilidad  h ∈ {0, 0,05, 0,1, 0,2, 0,3, 0,4} y K ∈ {3, 5, 7, 9}, uno a la vez desde la
                configuración elegida
  semillas      semillas 42-46 para el EJUL y para XGBoost
  ruido         ruido multiplicativo en la razón de entrenamiento, R' = R·exp(ε), ε ~ N(0, σ²),
                σ ∈ {0,1, 0,3, 0,5}; la etiqueta de entrenamiento se recalcula con R' para ambos
                modelos y se evalúa con la etiqueta limpia de V
  desbalance    submuestreo de positivos del entrenamiento hasta tasas de 15 % y 8 %
  muestra       submuestras de S de 5, 10 y 20 mil filas
Uso: python scripts/19_robustez.py <experimento> <pliegue> [n_jobs]
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config as C  # noqa: E402
import datos as D  # noqa: E402
import evaluacion as EV  # noqa: E402
import experimento as X  # noqa: E402
import modelos as M  # noqa: E402
import preprocesamiento as P  # noqa: E402
from ejul import EnsambleUmbralesLatentes  # noqa: E402

RES = RAIZ / "data" / "resultados"
SAL = RES / "robustez"


def importar_12():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m12", RAIZ / "scripts" / "12_modelo_original.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def metricas(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {"auc": roc_auc_score(y, p), "auc_pr": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p), "ece": EV.ece(y, p)}


def main(experimento, pliegue, n_jobs=2):
    warnings.filterwarnings("ignore")
    C.fijar_semillas(C.SEMILLA)
    m12 = importar_12()
    res = joblib.load(RES / "original" / f"{pliegue}.joblib")["resumen"]
    cfg = dict(res["params"])
    panel = D.cargar_panel()
    G = m12.codigos_grupo(panel)
    razon = pd.Series(panel["ratio_t1"].to_numpy(), index=panel["fila"].to_numpy())
    S, V = X.cargar_pliegue(pliegue)
    prep = P.preprocesador("arboles", "clf")
    X_S, X_V = m12.matriz(prep, S, G, True), m12.matriz(prep, V, G)
    y_S, y_V = S[C.OBJ_CLF].to_numpy(), V[C.OBJ_CLF].to_numpy()
    R_S = razon.loc[S["fila"]].to_numpy()
    T_S = S["semana_t1"].to_numpy()
    cod = res.get("codificacion", "cruzada")
    rng = np.random.default_rng(C.SEMILLA)

    def xgb(semilla=C.SEMILLA):
        b = M.construir(M.catalogo("clf")["XGBoost"], res["params_base"], semilla=semilla)
        b.set_params(n_jobs=n_jobs)
        return b

    def ajustar(Xs, ys, Rs, semilla=C.SEMILLA, idx=None, **cambios):
        t0 = time.perf_counter()
        Ts = T_S if idx is None else T_S[idx]
        pe = EnsambleUmbralesLatentes(estimador_base=xgb(semilla), columnas_grupo=(-3, -2, -1), random_state=semilla,
                                      codificacion=cod, **{**cfg, **cambios}).fit(Xs, ys, latente=Rs, tiempos=Ts).predict_proba(X_V)[:, 1]
        t1 = time.perf_counter()
        px = xgb(semilla).fit(Xs[:, :-3], ys).predict_proba(X_V[:, :-3])[:, 1]
        return {"ejul": metricas(y_V, pe), "xgb": metricas(y_V, px), "segundos": t1 - t0}

    filas = []

    def registrar(condicion, valor, r):
        for modelo in ("ejul", "xgb"):
            filas.append({"experimento": experimento, "pliegue": pliegue, "condicion": condicion, "valor": valor,
                          "modelo": modelo, **r[modelo]})
        print(pliegue, condicion, valor, round(r["ejul"]["auc"] - r["xgb"]["auc"], 5), flush=True)

    if experimento == "sensibilidad":
        for h in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]:
            r = ajustar(X_S, y_S, R_S, h=h)
            registrar("h", h, r)
        for K in [3, 5, 7, 9]:
            r = ajustar(X_S, y_S, R_S, n_umbrales=K)
            registrar("K", K, r)
    elif experimento == "semillas":
        for s in range(42, 47):
            registrar("semilla", s, ajustar(X_S, y_S, R_S, semilla=s))
    elif experimento == "ruido":
        for sigma in [0.1, 0.3, 0.5]:
            Rr = R_S * np.exp(rng.normal(0, sigma, len(R_S)))
            yr = (Rr < C.UMBRAL_INTERRUPCION).astype(int)
            registrar("sigma", sigma, ajustar(X_S, yr, Rr))
    elif experimento == "desbalance":
        pos = np.flatnonzero(y_S == 1)
        neg = np.flatnonzero(y_S == 0)
        for tasa in [0.15, 0.08]:
            n_pos = int(tasa / (1 - tasa) * len(neg))
            idx = np.sort(np.r_[neg, rng.choice(pos, n_pos, replace=False)])
            registrar("tasa_positivos", tasa, ajustar(X_S[idx], y_S[idx], R_S[idx], idx=idx))
    elif experimento == "muestra":
        for n in [5_000, 10_000, 20_000]:
            idx = np.sort(rng.choice(len(y_S), n, replace=False))
            registrar("n", n, ajustar(X_S[idx], y_S[idx], R_S[idx], idx=idx))
    else:
        raise ValueError(experimento)
    SAL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_parquet(SAL / f"{experimento}_{pliegue}.parquet", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
