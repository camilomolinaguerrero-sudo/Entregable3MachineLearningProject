import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
import experimento as X, modelos as M, preprocesamiento as P, config as C, datos as D, evaluacion as EV
from ejul import EnsambleUmbralesLatentes, logistica_newton, rasgos_calibracion
m = pd.read_parquet(sys.argv[2])
panel = D.cargar_panel().set_index("fila")
G = pd.DataFrame({"c": pd.factorize(panel["mercado"].astype(str) + "|" + panel["cod_mpio"].astype(str))[0],
                  "m": pd.factorize(panel["cod_mpio"].astype(str))[0], "d": pd.factorize(panel["depto_origen"].astype(str))[0]}, index=panel.index)
ext = m[(m.tarea=="clf")&(m.combinacion=="XGBoost | Sin balanceo | Random Search")].set_index("pliegue")
pe = pd.read_parquet("data/resultados/pred_externo_clf.parquet")
configs = json.loads(sys.argv[3])
def met(y, p):
    return {"auc": roc_auc_score(y, p), "pr": average_precision_score(y, p), "brier": brier_score_loss(y, p), "ll": log_loss(y, p), "ece": EV.ece(y, p)}
filas = []
for pl in ["k1","k2","k3","k4"]:
    S, V = X.cargar_pliegue(pl)
    params = json.loads(ext.loc[pl, "params_json"])
    prep = P.preprocesador("arboles","clf")
    ys = S[C.OBJ_CLF].values; yv = V[C.OBJ_CLF].values
    Xs = np.c_[prep.fit_transform(S, ys), G.loc[S["fila"]].values]; Xv = np.c_[prep.transform(V), G.loc[V["fila"]].values]
    p_ = Xs.shape[1]; cg = (p_-3, p_-2, p_-1)
    r_s = panel.loc[S["fila"], "ratio_t1"].values
    base_est = M.construir(M.catalogo("clf")["XGBoost"], params); base_est.set_params(n_jobs=8)
    xg = pe[pe.pliegue==pl].set_index("fila").loc[V["fila"], "XGBoost | Sin balanceo | Random Search"].values
    filas.append({"pliegue": pl, "config": "XGB", "cal": "-", **met(yv, xg)})
    for nombre, cfg in configs.items():
        for tipo, plie in [("rand", None), ("temp", D.pliegues_internos(S["semana_t1"]))]:
            mod = EnsambleUmbralesLatentes(estimador_base=base_est, columnas_grupo=cg, **cfg).fit(Xs, ys, latente=r_s, pliegues=plie)
            raw = mod.probabilidad_cruda(Xv)
            if tipo == "rand":
                filas.append({"pliegue": pl, "config": nombre, "cal": "crudo", **met(yv, raw)})
            ok = np.isfinite(mod.oof_crudo_)
            for cal in ["platt", "beta"]:
                w = logistica_newton(rasgos_calibracion(mod.oof_crudo_[ok], cal), ys[ok])
                pc = expit(w[0] + rasgos_calibracion(raw, cal) @ w[1:])
                filas.append({"pliegue": pl, "config": nombre, "cal": f"{cal}_{tipo}", **met(yv, pc)})
        print(pl, nombre, flush=True)
df = pd.DataFrame(filas); df.to_csv(sys.argv[4], index=False)
r = df.groupby(["config", "cal"])[["auc","pr","brier","ll","ece"]].mean()
print(r.round(5).to_string())
print(df.pivot_table(index=["config","cal"], columns="pliegue", values="ece").round(4).to_string())
