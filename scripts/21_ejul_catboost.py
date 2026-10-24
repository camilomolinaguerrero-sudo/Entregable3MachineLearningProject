"""Transferibilidad del mecanismo: EJUL con CatBoost como clasificador base.

El EJUL no depende del clasificador base. Aquí se reemplaza XGBoost por CatBoost con los
hiperparámetros que eligió el bucle interno de CatBoost en cada pliegue (script 16) y se conservan los
hiperparámetros propios del EJUL elegidos con XGBoost (script 12); no hay búsqueda adicional, de modo que
el presupuesto de ajuste del EJUL-CatBoost es menor que el de CatBoost solo. Se compara con CatBoost en
los mismos pliegues.
Uso: python scripts/21_ejul_catboost.py <pliegue> [n_jobs]
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config as C  # noqa: E402
import datos as D  # noqa: E402
import evaluacion as EV  # noqa: E402
import experimento as X  # noqa: E402
import preprocesamiento as P  # noqa: E402
from ejul import EnsambleUmbralesLatentes  # noqa: E402

RES = RAIZ / "data" / "resultados"


def main(pl, n_jobs=2):
    warnings.filterwarnings("ignore")
    import importlib.util
    from catboost import CatBoostClassifier
    spec = importlib.util.spec_from_file_location("m12", RAIZ / "scripts" / "12_modelo_original.py")
    m12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m12)
    ejul = joblib.load(RES / "original" / f"{pl}.joblib")["resumen"]
    cat = joblib.load(RES / "sota" / f"CatBoost_{pl}.joblib")
    panel = D.cargar_panel()
    G = m12.codigos_grupo(panel)
    razon = pd.Series(panel["ratio_t1"].to_numpy(), index=panel["fila"].to_numpy())
    S, V = X.cargar_pliegue(pl)
    prep = P.preprocesador("arboles", "clf")
    X_S, X_V = m12.matriz(prep, S, G, True), m12.matriz(prep, V, G)
    y_S, y_V = S[C.OBJ_CLF].to_numpy(), V[C.OBJ_CLF].to_numpy()
    base = CatBoostClassifier(iterations=300, random_seed=C.SEMILLA, thread_count=n_jobs, verbose=0,
                              allow_writing_files=False, **cat["resumen"]["params"])
    t0 = time.perf_counter()
    mod = EnsambleUmbralesLatentes(estimador_base=base, columnas_grupo=(-3, -2, -1), random_state=C.SEMILLA,
                                   codificacion=ejul["codificacion"], **ejul["params"])
    mod.fit(X_S, y_S, latente=razon.loc[S["fila"]].to_numpy(), tiempos=S["semana_t1"].to_numpy())
    p = mod.predict_proba(X_V)[:, 1]
    ok = np.isfinite(mod.oof_)
    umbral = EV.umbral_f1(y_S[ok], mod.oof_[ok])
    r = EV.metricas_clf(y_V, p, umbral, p)
    r["segundos"] = time.perf_counter() - t0
    pc = pd.Series(cat["pred"]["s"], index=cat["pred"]["fila"]).loc[V["fila"]].to_numpy()
    rc = EV.metricas_clf(y_V, pc, cat["resumen"]["umbral"], pc)
    out = {"pliegue": pl, "ejul_catboost": r, "catboost": rc, "pred": {"fila": V["fila"].to_numpy(), "s": p.astype(np.float32)}}
    (RES / "transferencia").mkdir(exist_ok=True)
    joblib.dump(out, RES / "transferencia" / f"{pl}.joblib", compress=3)
    print(pl, {k: round(r[k] - rc[k], 5) for k in ["auc", "auc_pr", "brier", "log_loss", "ece", "f1"]}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
