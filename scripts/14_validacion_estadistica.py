"""Bloque D: verificación estadística de la superioridad del modelo original (AJ-KL; EJUL con MODELO_ORIGINAL=ejul).

Universo: el modelo original y la mejor combinación de cada uno de los 7 modelos (Bloque B), con las mismas
particiones externas, la misma semilla y el mismo preprocesamiento.

Pruebas (ruta de clasificación con dependencia temporal, guía sección 6):
1. Friedman con corrección de Iman-Davenport sobre 48 bloques mensuales del bucle externo
   (k1-k4), con p por permutación restringida al pliegue y post hoc de Nemenyi (diagrama CD).
   Procedimiento global (controla el error de familia sobre los 8 modelos).
2. Wilcoxon de rangos con signo pareado (modelo original frente a cada benchmark, 48 meses) con Holm.
3. DeLong sobre la prueba 2025 con el error estándar por bootstrap de bloques de semana (la
   versión clásica supone independencia entre filas), Holm sobre las 7 comparaciones.
4. Model Confidence Set (Hansen et al., 2011) sobre las pérdidas semanales (Brier y log-loss) de
   2025: control de data snooping.
5. Diebold-Mariano con corrección HLN sobre las diferencias semanales de pérdida en 2025.
6. Prueba t corregida de Nadeau-Bengio sobre los 5 pliegues externos (k1-k4 y final).
7. Prueba bayesiana de rangos con signo con región de equivalencia práctica (ROPE).
Tamaños del efecto: delta de Cliff pareado (dominancia) y no pareado, d de Cohen pareado.
Intervalos: BCa con bootstrap de bloques de semana (prueba 2025) para AUC, AUC-PR, Brier,
log-loss y ECE.
"""
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config as C  # noqa: E402
import estadistica as E  # noqa: E402
import evaluacion as EV  # noqa: E402

RES = RAIZ / "data" / "resultados"
# El modelo original se elige con la variable de entorno MODELO_ORIGINAL: "ajkl" (por defecto) o "ejul"
MODELO = os.environ.get("MODELO_ORIGINAL", "ajkl")
ORIG, DIR_ORIG, SAL = {
    "ajkl": ("AJ-KL (original) | Sin balanceo | Bayesiana (TPE)", RES / "ajkl", RES / "validacion"),
    "ejul": ("EJUL (original) | Sin balanceo | Bayesiana (TPE)", RES / "original", RES / "validacion_ejul"),
}[MODELO]
B = 2000
ROPE = 0.01   # escala de ΔAUC del curso: diferencias menores que 0,01 se consideran triviales


def cargar():
    sel = pd.read_csv(RES / "seleccion_mejores.csv")
    bench = sel["combinacion"].tolist()
    ext = pd.read_parquet(RES / "pred_externo_clf.parquet")
    pru = pd.read_parquet(RES / "pred_prueba_clf.parquet")
    for pl in ["k1", "k2", "k3", "k4"]:
        p = joblib.load(DIR_ORIG / f"{pl}.joblib")["pred"]
        s = pd.Series(p["s"], index=p["fila"])
        m = ext["pliegue"] == pl
        ext.loc[m, ORIG] = s.reindex(ext.loc[m, "fila"]).to_numpy()
    p = joblib.load(DIR_ORIG / "final.joblib")["pred"]
    pru[ORIG] = pd.Series(p["s"], index=p["fila"]).reindex(pru["fila"]).to_numpy()
    for m in ["LightGBM", "CatBoost"]:
        for pl in ["k1", "k2", "k3", "k4"]:
            q = joblib.load(RES / "sota" / f"{m}_{pl}.joblib")["pred"]
            msk = ext["pliegue"] == pl
            ext.loc[msk, m] = pd.Series(q["s"], index=q["fila"]).reindex(ext.loc[msk, "fila"]).to_numpy()
        q = joblib.load(RES / "sota" / f"{m}_final.joblib")["pred"]
        pru[m] = pd.Series(q["s"], index=q["fila"]).reindex(pru["fila"]).to_numpy()
    assert ext[ORIG].notna().all() and pru[ORIG].notna().all()
    # probabilidades: la SVM usa su escalado de Platt (columna "| prob")
    prob = {c: (c + " | prob" if (c + " | prob") in ext.columns else c) for c in bench + [ORIG, "LightGBM", "CatBoost"]}
    return sel, bench, ext, pru, prob


def por_bloque(df, modelos, col_bloque, prob):
    """Métricas por bloque (mes): AUC, AUC-PR, Brier y log-loss."""
    filas = []
    for b, g in df.groupby(col_bloque):
        y = g[C.OBJ_CLF].to_numpy()
        for m in modelos:
            s = g[m].to_numpy(float)
            p = np.clip(g[prob[m]].to_numpy(float), 1e-7, 1 - 1e-7)
            filas.append({"bloque": str(b), "modelo": m, "auc": roc_auc_score(y, s), "auc_pr": average_precision_score(y, s),
                          "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1])})
    return pd.DataFrame(filas)


def bayes_signed_rank(d, rope, B=20000, semilla=42):
    """Prueba bayesiana de rangos con signo (Benavoli et al., 2014, 2017).

    Proceso de Dirichlet con prior de concentración 1 en el pseudo-dato z0 = 0. En cada muestra
    w ~ Dir(1, ..., 1, 1) se calcula la probabilidad de que la diferencia media (por pares de
    Walsh) caiga a la izquierda de −rope, dentro de [−rope, rope] o a la derecha de rope.
    """
    d = np.r_[0.0, np.asarray(d, float)]
    n = len(d)
    S = d[:, None] + d[None, :]                       # promedios de Walsh × 2
    izq = (S < -2 * rope).astype(float)
    der = (S > 2 * rope).astype(float)
    rng = np.random.default_rng(semilla)
    cuenta = np.zeros(3)
    for _ in range(B):
        w = rng.dirichlet(np.ones(n))
        ww = np.outer(w, w)
        pi = (ww * izq).sum(), (ww * der).sum()
        theta = np.array([pi[0], 1 - pi[0] - pi[1], pi[1]])
        cuenta[np.argmax(theta)] += 1
    return {"p_izq": cuenta[0] / B, "p_rope": cuenta[1] / B, "p_der": cuenta[2] / B}


def nadeau_bengio(d, n_entr, n_eval):
    """t corregida de Nadeau y Bengio (2003): var = (1/J + n_eval/n_entr) s²."""
    d = np.asarray(d, float)
    J = len(d)
    var = (1 / J + n_eval / n_entr) * d.var(ddof=1)
    t = d.mean() / np.sqrt(var)
    return {"t": t, "gl": J - 1, "p": 2 * stats.t.sf(abs(t), J - 1), "media": d.mean()}


def bca_metricas(pru, a, b, prob, semanas):
    """Intervalos BCa (bloques de semana) de la diferencia a − b en cinco métricas sobre 2025."""
    y = pru[C.OBJ_CLF].to_numpy(float)
    sa, sb = pru[a].to_numpy(float), pru[b].to_numpy(float)
    pa = np.clip(pru[prob[a]].to_numpy(float), 1e-7, 1 - 1e-7)
    pb = np.clip(pru[prob[b]].to_numpy(float), 1e-7, 1 - 1e-7)
    Aa, Ab = E.AUCPonderado(y, sa), E.AUCPonderado(y, sb)
    la = -(y * np.log(pa) + (1 - y) * np.log(1 - pa)); lb = -(y * np.log(pb) + (1 - y) * np.log(1 - pb))
    ba = (pa - y) ** 2; bb = (pb - y) ** 2
    orden_a, orden_b = np.argsort(-sa, kind="mergesort"), np.argsort(-sb, kind="mergesort")

    def ap(w, s_orden):
        yy, ww = y[s_orden], w[s_orden]
        tp = np.cumsum(ww * yy)
        prec = tp / np.maximum(np.cumsum(ww), 1e-12)
        return float(np.sum(prec * ww * yy) / max(tp[-1], 1e-12))

    bins = np.linspace(0, 1, 16)

    def ece_w(w, p):
        idx = np.clip(np.digitize(p, bins) - 1, 0, 14)
        nw = np.bincount(idx, weights=w, minlength=15)
        dy = np.bincount(idx, weights=w * y, minlength=15)
        dp = np.bincount(idx, weights=w * p, minlength=15)
        return float(np.sum(np.abs(dy - dp)) / w.sum())

    estad = {
        "auc": lambda w: Aa.auc(w) - Ab.auc(w),
        "auc_pr": lambda w: ap(w, orden_a) - ap(w, orden_b),
        "brier": lambda w: float(np.sum(w * (ba - bb)) / w.sum()),
        "log_loss": lambda w: float(np.sum(w * (la - lb)) / w.sum()),
        "ece": lambda w: ece_w(w, pa) - ece_w(w, pb),
    }
    return {k: E.bca_bloques(f, semanas, B=B) for k, f in estad.items()}


def main():
    warnings.filterwarnings("ignore")
    SAL.mkdir(parents=True, exist_ok=True)
    sel, bench, ext, pru, prob = cargar()
    modelos = bench + [ORIG]
    corto = {c: c.split(" | ")[0] for c in modelos}
    ext["mes"] = pd.to_datetime(ext["semana_t1"]).dt.to_period("M")
    pru["mes"] = pd.to_datetime(pru["semana_t1"]).dt.to_period("M")
    blo = por_bloque(ext, modelos, "mes", prob)
    blo.to_parquet(SAL / "metricas_mensuales_externo.parquet", index=False)
    blo25 = por_bloque(pru, modelos, "mes", prob)
    blo25.to_parquet(SAL / "metricas_mensuales_2025.parquet", index=False)
    pliegue_mes = ext.groupby("mes")["pliegue"].first().astype(str)
    filas_pruebas, resumen = [], {}

    # ---------------------------------------------------------------- 1. Friedman + Nemenyi
    tabla = blo.pivot(index="bloque", columns="modelo", values="auc")[modelos]
    fr = E.friedman(tabla)
    perm = E.friedman_permutacion(tabla, pliegue_mes.reindex(tabla.index.map(pd.Period)).to_numpy(), B=10000)
    cd = E.diferencia_critica(len(modelos), len(tabla))
    nem = E.nemenyi(tabla)
    nem.to_parquet(SAL / "nemenyi_auc.parquet")
    resumen["friedman"] = {"chi2": fr["chi2"], "p": fr["p"], "F_ID": fr["F_ID"], "p_F": fr["p_F"], "p_perm": perm["p_perm"],
                           "N": fr["N"], "k": fr["k"], "CD": cd, "rangos": fr["rangos"].to_dict()}
    filas_pruebas.append({"prueba": "Friedman (Iman-Davenport) + permutación por pliegue", "comparacion": "8 modelos",
                          "h0": "los 8 modelos tienen el mismo AUC mensual (mismos rangos)",
                          "supuestos": "bloques mensuales; dependencia dentro del pliegue tratada con permutación restringida",
                          "estadistico": fr["F_ID"], "p_crudo": max(fr["p_F"], perm["p_perm"]), "p_ajustado": max(fr["p_F"], perm["p_perm"]),
                          "efecto": f"rango {corto[ORIG]} {fr['rangos'][ORIG]:.2f} (CD = {cd:.2f})", "familia": "global"})
    for b in bench:
        dif_r = fr["rangos"][b] - fr["rangos"][ORIG]
        filas_pruebas.append({"prueba": "Nemenyi (post hoc)", "comparacion": f"{corto[ORIG]} vs {corto[b]}",
                              "h0": "rangos medios iguales", "supuestos": "post hoc de Friedman; ajuste incluido en q de Tukey",
                              "estadistico": dif_r, "p_crudo": float(nem.loc[ORIG, b]), "p_ajustado": float(nem.loc[ORIG, b]),
                              "efecto": f"Δrango = {dif_r:.2f}", "familia": "Nemenyi"})

    # ---------------------------------------------------------------- 2. Wilcoxon + efectos
    filas_w = []
    for metrica, mayor in [("auc", True), ("auc_pr", True), ("brier", False), ("log_loss", False)]:
        t = blo.pivot(index="bloque", columns="modelo", values=metrica)
        for b in bench:
            d = (t[ORIG] - t[b]) if mayor else (t[b] - t[ORIG])           # > 0 = modelo original mejor
            w = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
            cp, mag_p = E.dominancia_pareada(d, np.zeros(len(d)))
            cu, mag_u = E.delta_cliff(t[ORIG], t[b]) if mayor else E.delta_cliff(-t[ORIG], -t[b])
            dz, mag_d = E.d_cohen_pareado(d)
            ic = E.ic_bootstrap_estacionario(d.to_numpy(), B=B)
            by = bayes_signed_rank(d.to_numpy(), ROPE if metrica in ("auc", "auc_pr") else 0.0005)
            filas_w.append({"metrica": metrica, "benchmark": b, "media_dif": d.mean(), "ic_inf": ic["ic_inf"], "ic_sup": ic["ic_sup"],
                            "meses_gana": int((d > 0).sum()), "meses": len(d), "W": w.statistic, "p": w.pvalue,
                            "cliff_pareado": cp, "mag_cliff_pareado": mag_p, "cliff": cu, "mag_cliff": mag_u,
                            "d_cohen": dz, "mag_d": mag_d, **{f"bayes_{k}": v for k, v in by.items()}})
    W = pd.DataFrame(filas_w)
    W["p_holm"] = W.groupby("metrica")["p"].transform(lambda p: E.ajustar_p(p, "holm"))
    W.to_parquet(SAL / "wilcoxon_mensual.parquet", index=False)
    for r in W[W.metrica == "auc"].itertuples():
        filas_pruebas.append({"prueba": "Wilcoxon pareado (Holm)", "comparacion": f"{corto[ORIG]} vs {corto[r.benchmark]}",
                              "h0": "mediana de ΔAUC mensual = 0", "supuestos": "diferencias simétricas; 48 meses",
                              "estadistico": r.W, "p_crudo": r.p, "p_ajustado": r.p_holm,
                              "efecto": f"Cliff pareado {r.cliff_pareado:.2f} ({r.mag_cliff_pareado}); d_z {r.d_cohen:.2f}",
                              "familia": "Wilcoxon AUC"})

    # ---------------------------------------------------------------- 3. DeLong + BCa (2025)
    y = pru[C.OBJ_CLF].to_numpy()
    semanas = pru["semana_t1"].astype(str).to_numpy()
    filas_d = []
    for b in bench:
        dl = E.delong_prueba(y, pru[ORIG], pru[b])
        ics = bca_metricas(pru, ORIG, b, prob, semanas)
        z_b = dl["delta"] / ics["auc"]["se"]
        filas_d.append({"benchmark": b, "auc_orig": dl["auc1"], "auc_bench": dl["auc2"], "delta": dl["delta"],
                        "p_delong_iid": dl["p"], "se_delong": (dl["delta_sup"] - dl["delta"]) / 1.959964,
                        "se_bloques": ics["auc"]["se"], "z_bloques": z_b, "p_delong_bloques": 2 * stats.norm.sf(abs(z_b)),
                        **{f"{k}_dif": v["estimado"] for k, v in ics.items()},
                        **{f"{k}_ic_inf": v["ic_inf"] for k, v in ics.items()},
                        **{f"{k}_ic_sup": v["ic_sup"] for k, v in ics.items()},
                        **{f"{k}_p_boot": v["p_boot"] for k, v in ics.items()}})
        print("DeLong", corto[b], round(dl["delta"], 5), flush=True)
    Dl = pd.DataFrame(filas_d)
    Dl["p_holm"] = E.ajustar_p(Dl["p_delong_bloques"], "holm")
    Dl.to_parquet(SAL / "delong_2025.parquet", index=False)
    for r in Dl.itertuples():
        filas_pruebas.append({"prueba": "DeLong con EE por bloques de semana (Holm)", "comparacion": f"{corto[ORIG]} vs {corto[r.benchmark]}",
                              "h0": "AUC iguales en 2025", "supuestos": "normalidad asintótica de ΔAUC; dependencia semanal en el EE",
                              "estadistico": r.z_bloques, "p_crudo": r.p_delong_bloques, "p_ajustado": r.p_holm,
                              "efecto": f"ΔAUC {r.delta:+.4f} [{r.auc_ic_inf:+.4f}; {r.auc_ic_sup:+.4f}]", "familia": "DeLong 2025"})

    # ---------------------------------------------------------------- 4. MCS y 5. Diebold-Mariano (2025)
    pru["semana"] = pru["semana_t1"].astype(str)
    perd = {}
    for nombre, f in [("brier", lambda yy, p: (p - yy) ** 2), ("log_loss", lambda yy, p: -(yy * np.log(p) + (1 - yy) * np.log(1 - p)))]:
        L = pd.DataFrame({corto[m]: f(pru[C.OBJ_CLF].to_numpy(float), np.clip(pru[prob[m]].to_numpy(float), 1e-7, 1 - 1e-7)) for m in modelos})
        L["semana"] = pru["semana"].to_numpy()
        perd[nombre] = L.groupby("semana").mean()
    mcs = {k: E.model_confidence_set(v, alfa=0.10, B=B) for k, v in perd.items()}
    for k, v in mcs.items():
        v.to_parquet(SAL / f"mcs_{k}_2025.parquet")
        filas_pruebas.append({"prueba": f"Model Confidence Set ({k} semanal, α = 0,10)", "comparacion": "8 modelos",
                              "h0": "igual capacidad predictiva de los modelos del conjunto",
                              "supuestos": "pérdidas semanales estacionarias; bootstrap estacionario",
                              "estadistico": float(v.loc[corto[ORIG], "perdida_media"]),
                              "p_crudo": float(v.drop(corto[ORIG])["p_mcs"].max()), "p_ajustado": float(v.drop(corto[ORIG])["p_mcs"].max()),
                              "efecto": "conjunto: " + ", ".join(v.index[v.en_conjunto]), "familia": "global (data snooping)"})
    filas_dm = []
    for k, L in perd.items():
        for b in bench:
            dm = E.diebold_mariano(L[corto[b]].to_numpy(), L[corto[ORIG]].to_numpy())
            ic = E.ic_bootstrap_estacionario((L[corto[b]] - L[corto[ORIG]]).to_numpy(), B=B)
            filas_dm.append({"perdida": k, "benchmark": b, **dm, "ic_inf": ic["ic_inf"], "ic_sup": ic["ic_sup"],
                             "semanas_gana": int((L[corto[b]] > L[corto[ORIG]]).sum()), "semanas": len(L)})
    DM = pd.DataFrame(filas_dm)
    DM["p_holm"] = DM.groupby("perdida")["p"].transform(lambda p: E.ajustar_p(p, "holm"))
    DM.to_parquet(SAL / "diebold_mariano_2025.parquet", index=False)
    for r in DM[DM.perdida == "log_loss"].itertuples():
        filas_pruebas.append({"prueba": "Diebold-Mariano HLN, log-loss semanal (Holm)", "comparacion": f"{corto[ORIG]} vs {corto[r.benchmark]}",
                              "h0": "igual log-loss esperada", "supuestos": "diferencial de pérdida estacionario; varianza HAC",
                              "estadistico": r.DM_HLN, "p_crudo": r.p, "p_ajustado": r.p_holm,
                              "efecto": f"gana {r.semanas_gana}/{r.semanas} semanas", "familia": "DM 2025"})

    # ---------------------------------------------------------------- 6. Nadeau-Bengio (5 pliegues)
    maestra = pd.read_parquet(RES / "maestra.parquet")
    ejul_pl = {pl: joblib.load(DIR_ORIG / f"{pl}.joblib")["resumen"]["auc"] for pl in ["k1", "k2", "k3", "k4", "final"]}
    filas_nb = []
    for b in bench:
        mb = maestra[(maestra.tarea == "clf") & (maestra.combinacion == b)].set_index("pliegue")["auc"]
        d = np.array([ejul_pl[pl] - mb[pl] for pl in ejul_pl])
        nb = nadeau_bengio(d, n_entr=C.N_MUESTRA, n_eval=int(len(pru)))
        filas_nb.append({"benchmark": b, **nb, "pliegues_gana": int((d > 0).sum())})
    NB = pd.DataFrame(filas_nb)
    NB["p_holm"] = E.ajustar_p(NB["p"], "holm")
    NB.to_parquet(SAL / "nadeau_bengio.parquet", index=False)

    # ---------------------------------------------------------------- comparadores del estado del arte
    sota = ["LightGBM", "CatBoost"]
    blo_s = por_bloque(ext, [ORIG] + sota, "mes", prob)
    ts = blo_s.pivot(index="bloque", columns="modelo", values="auc")
    filas_s = []
    for b in sota:
        d = ts[ORIG] - ts[b]
        w = stats.wilcoxon(d)
        dl = E.delong_prueba(y, pru[ORIG], pru[b])
        ics = bca_metricas(pru, ORIG, b, prob, semanas)
        z_b = dl["delta"] / ics["auc"]["se"]
        filas_s.append({"comparador": b, "auc_ext_orig": float(ts[ORIG].mean()), "auc_ext_comp": float(ts[b].mean()),
                        "meses_gana": int((d > 0).sum()), "p_wilcoxon": w.pvalue, "delta_2025": dl["delta"],
                        "ic_inf": ics["auc"]["ic_inf"], "ic_sup": ics["auc"]["ic_sup"], "p_delong_bloques": 2 * stats.norm.sf(abs(z_b)),
                        "brier_dif": ics["brier"]["estimado"], "log_loss_dif": ics["log_loss"]["estimado"], "ece_dif": ics["ece"]["estimado"]})
    S_ = pd.DataFrame(filas_s)
    S_["p_wilcoxon_holm"] = E.ajustar_p(S_["p_wilcoxon"], "holm")
    S_["p_delong_holm"] = E.ajustar_p(S_["p_delong_bloques"], "holm")
    S_.to_parquet(SAL / "comparadores_sota.parquet", index=False)
    print(S_.round(5).to_string())

    T = pd.DataFrame(filas_pruebas)
    T["rechaza_h0"] = T["p_ajustado"] < np.where(T["prueba"].str.startswith("Model Confidence"), 0.10, 0.05)
    T.to_parquet(SAL / "tabla_pruebas.parquet", index=False)
    (SAL / "resumen.json").write_text(json.dumps(resumen, indent=2, default=float, ensure_ascii=False))
    print(T[["prueba", "comparacion", "p_ajustado", "efecto", "rechaza_h0"]].to_string())
    print(W[W.metrica == "auc"][["benchmark", "media_dif", "meses_gana", "p_holm", "cliff_pareado", "cliff", "d_cohen",
                                 "bayes_p_rope", "bayes_p_der"]].to_string())
    print(NB.to_string())


if __name__ == "__main__":
    main()
