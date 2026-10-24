"""Comparación complementaria con los modelos de producción (Bloque B).

Los artefactos (los 7 benchmarks del curso, LightGBM, CatBoost y el AJ-KL) se reentrenaron con las 622.244 filas del periodo de entrenamiento y se evalúan una
sola vez en las 103.466 filas de 2025. Para el AJ-KL frente a cada benchmark se calculan:
  * DeLong con error estándar por bootstrap de bloques de semana (Holm sobre las 9 comparaciones);
  * McNemar exacto sobre las decisiones con el umbral de cada modelo (Holm);
  * intervalos BCa por bloques de semana de la diferencia en AUC, AUC-PR, Brier, log-loss y ECE.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))
import config as C  # noqa: E402
import estadistica as E  # noqa: E402

RES = RAIZ / "data" / "resultados"
SAL = RES / "validacion"


def main():
    warnings.filterwarnings("ignore")
    import importlib.util
    spec = importlib.util.spec_from_file_location("v", RAIZ / "scripts" / "14_validacion_estadistica.py")
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    pa = pd.read_parquet(RES / "pred_prueba_artefactos.parquet")
    art = pd.read_csv(RES / "artefactos_resumen.csv").set_index("combinacion")
    art = art[~art.index.str.startswith("EJUL")]
    orig = [c for c in art.index if c.startswith("AJ-KL")][0]
    bench = [c for c in art.index if c != orig]
    y = pa[C.OBJ_CLF].to_numpy()
    semanas = pa["semana_t1"].astype(str).to_numpy()
    prob = {c: c for c in art.index}
    filas = []
    for b in bench:
        dl = E.delong_prueba(y, pa[orig], pa[b])
        ics = v.bca_metricas(pa, orig, b, prob, semanas)
        z = dl["delta"] / ics["auc"]["se"]
        ua, ub = art.loc[orig, "umbral_2025"], art.loc[b, "umbral_2025"]
        ca, cb = (pa[orig].to_numpy() >= ua) == y, (pa[b].to_numpy() >= ub) == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())          # el AJ-KL acierta y el otro no, y viceversa
        p_mc = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        filas.append({"benchmark": b, "auc_orig": dl["auc1"], "auc_bench": dl["auc2"], "delta": dl["delta"],
                      "p_delong_iid": dl["p"], "se_bloques": ics["auc"]["se"], "z_bloques": z,
                      "p_delong_bloques": 2 * stats.norm.sf(abs(z)), "mcnemar_orig_acierta": n01, "mcnemar_bench_acierta": n10,
                      "p_mcnemar": p_mc,
                      **{f"{k}_dif": x["estimado"] for k, x in ics.items()},
                      **{f"{k}_ic_inf": x["ic_inf"] for k, x in ics.items()},
                      **{f"{k}_ic_sup": x["ic_sup"] for k, x in ics.items()}})
        print(b.split(" | ")[0], round(dl["delta"], 5), round(filas[-1]["p_delong_bloques"], 5), flush=True)
    T = pd.DataFrame(filas)
    T["p_delong_holm"] = E.ajustar_p(T["p_delong_bloques"], "holm")
    T["p_mcnemar_holm"] = E.ajustar_p(T["p_mcnemar"], "holm")
    T.to_parquet(SAL / "produccion_2025.parquet", index=False)
    print(T[["benchmark", "delta", "auc_ic_inf", "auc_ic_sup", "p_delong_holm", "mcnemar_orig_acierta", "mcnemar_bench_acierta",
             "p_mcnemar_holm", "brier_dif", "log_loss_dif", "ece_dif"]].round(5).to_string())


if __name__ == "__main__":
    main()
