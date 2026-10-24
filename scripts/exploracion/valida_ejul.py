import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import experimento as X, modelos as M, preprocesamiento as P, config as C, datos as D, evaluacion as EV
from ejul import EnsambleUmbralesLatentes
m = pd.read_parquet(sys.argv[2])
panel = D.cargar_panel().set_index("fila")
cod_corr = pd.factorize(panel["mercado"].astype(str) + "|" + panel["cod_mpio"].astype(str))[0]
cod_mpio = pd.factorize(panel["cod_mpio"].astype(str))[0]
cod_dep = pd.factorize(panel["depto_origen"].astype(str))[0]
G = pd.DataFrame({"c": cod_corr, "m": cod_mpio, "d": cod_dep}, index=panel.index)
ext = m[(m.tarea=="clf")&(m.combinacion=="XGBoost | Sin balanceo | Random Search")].set_index("pliegue")
METR = ["auc","auc_pr","brier","log_loss","ece","f1","precision","recall","accuracy"]
configs = json.loads(sys.argv[3])
filas = []
for pl in ["k1","k2","k3","k4"]:
    S, V = X.cargar_pliegue(pl)
    params = json.loads(ext.loc[pl, "params_json"])
    prep = P.preprocesador("arboles","clf")
    ys = S[C.OBJ_CLF].values; yv = V[C.OBJ_CLF].values
    Xs = prep.fit_transform(S, ys); Xv = prep.transform(V)
    Xs = np.c_[Xs, G.loc[S["fila"]].values]; Xv = np.c_[Xv, G.loc[V["fila"]].values]
    p = Xs.shape[1]; cg = (p-3, p-2, p-1)
    r_s = panel.loc[S["fila"], "ratio_t1"].values
    plie = D.pliegues_internos(S["semana_t1"])
    base = {k: float(ext.loc[pl, k]) for k in METR}
    filas.append({"pliegue": pl, "modelo": "XGB (E2)", **base})
    for nombre, cfg in configs.items():
        t = time.time()
        base_est = M.construir(M.catalogo("clf")["XGBoost"], params); base_est.set_params(n_jobs=8)
        mod = EnsambleUmbralesLatentes(estimador_base=base_est, columnas_grupo=cg, **cfg).fit(Xs, ys, latente=r_s, pliegues=plie)
        pv = mod.predict_proba(Xv)[:, 1]
        ok = np.isfinite(mod.oof_)
        umb = EV.umbral_f1(ys[ok], mod.oof_[ok])
        r = EV.metricas_clf(yv, pv, umb, pv)
        filas.append({"pliegue": pl, "modelo": nombre, **{k: r[k] for k in METR}, "seg": time.time()-t})
        print(pl, nombre, {k: round(r[k]-base[k], 4) for k in METR}, round(time.time()-t), flush=True)
df = pd.DataFrame(filas)
df.to_csv(sys.argv[4], index=False)
res = df.groupby("modelo")[METR].mean()
print(res.round(5).to_string())
print((res - res.loc["XGB (E2)"]).round(5).to_string())
