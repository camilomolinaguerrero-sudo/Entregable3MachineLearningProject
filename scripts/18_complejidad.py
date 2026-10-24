"""Complejidad empírica del EJUL frente a su clasificador base (XGBoost).

Se mide el tiempo de entrenamiento y de predicción al variar
  * n ∈ {5, 10, 20, 40, 80} mil filas con todas las variables, y
  * p ∈ {25, 50, 75, 100} % de las variables con n = 20 mil,
con K = 5 umbrales, V = 3 pliegues de recalibración y 3 repeticiones por punto (se reporta la
mediana). La pendiente en escala log-log estima el exponente de n y p; la teoría (sección 5.4)
predice tiempo lineal en n y en p para XGBoost con histogramas y un factor K(V + 1) = 20 frente
al benchmark en el entrenamiento y K = 5 en la predicción.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config as C  # noqa: E402
import datos as D  # noqa: E402
import modelos as M  # noqa: E402
import preprocesamiento as P  # noqa: E402
from ejul import EnsambleUmbralesLatentes  # noqa: E402

SAL = RAIZ / "data" / "resultados" / "complejidad_ejul.parquet"
PARAMS = {"learning_rate": 0.026, "max_depth": 4, "min_child_weight": 43.6, "colsample_bytree": 0.69, "subsample": 0.89}


def medir(Xtr, ytr, rtr, Xte, n_jobs, rep=3, ttr=None):
    filas = []
    for modelo in ["XGBoost", "EJUL"]:
        tf, tp = [], []
        for r in range(rep):
            base = M.construir(M.catalogo("clf")["XGBoost"], PARAMS)
            base.set_params(n_jobs=n_jobs)
            if modelo == "XGBoost":
                m = base
                t0 = time.perf_counter(); m.fit(Xtr[:, :-3], ytr); tf.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); m.predict_proba(Xte[:, :-3]); tp.append(time.perf_counter() - t0)
            else:
                m = EnsambleUmbralesLatentes(estimador_base=base, columnas_grupo=(-3, -2, -1), h=0.2, n_umbrales=5,
                                             codificacion="temporal", random_state=C.SEMILLA + r)
                t0 = time.perf_counter(); m.fit(Xtr, ytr, latente=rtr, tiempos=ttr); tf.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); m.predict_proba(Xte); tp.append(time.perf_counter() - t0)
        filas.append({"modelo": modelo, "t_fit": float(np.median(tf)), "t_pred_1000": 1000 * float(np.median(tp)) / len(Xte)})
    return filas


def main(n_jobs=4):
    warnings.filterwarnings("ignore")
    panel = D.cargar_panel()
    train, _ = D.particion(panel)
    G = pd.DataFrame({"g1": pd.factorize(train["mercado"].astype(str) + "|" + train["cod_mpio"].astype(str))[0],
                      "g2": pd.factorize(train["cod_mpio"].astype(str))[0], "g3": pd.factorize(train["depto_origen"].astype(str))[0]})
    rng = np.random.default_rng(C.SEMILLA)
    idx = rng.permutation(len(train))[:100_000]
    tr_all, te = idx[:80_000], idx[80_000:90_000]
    prep = P.preprocesador("arboles", "clf")
    Xall = prep.fit_transform(train.iloc[idx[:90_000]], train[C.OBJ_CLF].to_numpy()[idx[:90_000]])
    Xall = np.c_[Xall, G.to_numpy()[idx[:90_000]]]
    y = train[C.OBJ_CLF].to_numpy()[idx[:90_000]]
    r = train["ratio_t1"].to_numpy()[idx[:90_000]]
    tt = train["semana_t1"].to_numpy()[idx[:90_000]]
    Xtr, ytr, rtr, ttr, Xte = Xall[:80_000], y[:80_000], r[:80_000], tt[:80_000], Xall[80_000:]
    filas = []
    for n in [5_000, 10_000, 20_000, 40_000, 80_000]:
        for f in medir(Xtr[:n], ytr[:n], rtr[:n], Xte, n_jobs, ttr=ttr[:n]):
            filas.append({"experimento": "n", "n": n, "p": Xtr.shape[1] - 3, **f})
        print("n", n, filas[-2:], flush=True)
    p_total = Xtr.shape[1] - 3
    for frac in [0.25, 0.5, 0.75, 1.0]:
        k = max(2, int(round(frac * p_total)))
        cols = list(rng.permutation(p_total)[:k]) + [p_total, p_total + 1, p_total + 2]
        for f in medir(Xtr[:20_000][:, cols], ytr[:20_000], rtr[:20_000], Xte[:, cols], n_jobs, ttr=ttr[:20_000]):
            filas.append({"experimento": "p", "n": 20_000, "p": k, **f})
        print("p", k, filas[-2:], flush=True)
    pd.DataFrame(filas).to_parquet(SAL, index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
