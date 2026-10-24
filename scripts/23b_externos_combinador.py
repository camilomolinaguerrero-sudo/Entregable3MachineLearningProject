"""Recalcula el combinador AJ-KL en los conjuntos externos con el optimizador de la versión 2.1.

Las predicciones de los 9 modelos base en cada tramo de evaluación ya están en data/resultados/externos/
(scripts/23_externos.py). El combinador del pliegue j (j = 2..5) se entrena con las predicciones de los
tramos anteriores, igual que en 23_externos.py; el pliegue 1 (historial con predicciones internas, que no se
guardaron) se excluye de la comparación del AJ-KL y se reporta aparte. Mismo TPE (24 evaluaciones, semilla 42).
Uso: python scripts/23b_externos_combinador.py
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config as C  # noqa: E402
import externos as EX  # noqa: E402
import optimizacion as O  # noqa: E402
from original_model import ApilamientoJerarquicoKL  # noqa: E402

spec = importlib.util.spec_from_file_location("m23", RAIZ / "scripts" / "23_externos.py")
m23 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m23)
SAL = RAIZ / "data" / "resultados" / "externos"


def main():
    warnings.filterwarnings("ignore")
    filas = []
    for conjunto in ["beijing", "bike", "bank"]:
        r = joblib.load(SAL / f"{conjunto}.joblib")
        pred = r["pred"]
        D = EX.CARGADORES[conjunto]()
        t_all = D["tiempo"]
        G_all = np.column_stack([pd.factorize(D["grupos"][c])[0] for c in ["n1", "n2", "n3"]])
        modelos = m23.CLASE + list(m23.SOTA)
        # recupera los códigos de la jerarquía de cada fila de evaluación por su posición en el tramo
        tramos = np.array_split(np.unique(t_all), 6)
        G_te = {j: G_all[np.isin(t_all, tramos[j])] for j in range(1, 6)}
        for j in range(2, 6):
            h = pred[pred.pliegue < j]
            v = pred[pred.pliegue == j]
            Gh = np.vstack([G_te[i] for i in range(1, j)])
            Xh, yh, th = m23.matriz(h[modelos].to_numpy(float), Gh), h["y"].to_numpy(), h["tiempo"].to_numpy(float)
            corte = np.quantile(th, 0.85)
            es_val = th > corte if (th > corte).any() and (th <= corte).any() else np.arange(len(th)) >= int(0.85 * len(th))
            C.fijar_semillas(C.SEMILLA)
            res = O.bayes_tpe(m23.ObjetivoAJKL(Xh, yh, th, es_val), m23.ESPACIO_AJKL, C.PRESUPUESTO, C.SEMILLA)
            hist = pd.DataFrame(res["historial"])
            i = int(hist["puntaje"].to_numpy().argmax())
            mejor, cal = dict(hist.loc[i, "params"]), bool(hist.loc[i, "calibrar"])
            Xv, yv = m23.matriz(v[modelos].to_numpy(float), G_te[j]), v["y"].to_numpy()
            p = ApilamientoJerarquicoKL(n_modelos=len(modelos), calibrar=cal, **mejor).fit(Xh, yh, tiempos=th).predict_proba(Xv)[:, 1]
            g = ApilamientoJerarquicoKL(n_modelos=len(modelos), calibrar=cal, **{**mejor, "profundidad": 1}).fit(
                Xh, yh, tiempos=th).predict_proba(Xv)[:, 1]
            fila = {"conjunto": conjunto, "nombre": r["nombre"], "pliegue": j, "n_eval": len(v), "tasa": float(yv.mean()),
                    "params_ajkl": {**mejor, "calibrar": cal}}
            for nombre, q in [("AJ-KL", p), ("AJ-KL global (profundidad 1)", g)] + [(m, v[m].to_numpy(float)) for m in modelos]:
                fila[nombre] = roc_auc_score(yv, q)
                fila[nombre + " | log_loss"] = log_loss(yv, np.clip(q, 1e-7, 1 - 1e-7))
                fila[nombre + " | brier"] = float(np.mean((q - yv) ** 2))
            filas.append(fila)
            print(conjunto, j, {k: round(x, 4) for k, x in fila.items() if isinstance(x, float) and "|" not in k}, flush=True)
    T = pd.DataFrame(filas)
    T.to_parquet(SAL / "comparacion_v21.parquet", index=False)


if __name__ == "__main__":
    main()
