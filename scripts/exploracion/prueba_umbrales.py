import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import TargetEncoder
from scipy.special import logit, expit
import experimento as X, modelos as M, preprocesamiento as P, config as C, datos as D
m = pd.read_parquet(sys.argv[2])
panel = D.cargar_panel().set_index("fila")
ext = m[(m.tarea=="clf")&(m.combinacion=="XGBoost | Sin balanceo | Random Search")]
grids = {"g3_estrecho": [0.4, 0.5, 0.6], "g5": [0.3,0.4,0.5,0.6,0.7], "g9_ancho": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.85,1.0], "g7_alto": [0.5,0.6,0.7,0.8,0.9,1.0,1.2]}
res = []
for pl in ["k1","k2","k3","k4"]:
    S, V = X.cargar_pliegue(pl)
    params = json.loads(ext[ext.pliegue==pl].params_json.iloc[0])
    espec = M.catalogo("clf")["XGBoost"]
    prep = P.preprocesador("arboles","clf")
    ys = S[C.OBJ_CLF].values; yv = V[C.OBJ_CLF].values
    Xs = prep.fit_transform(S, ys); Xv = prep.transform(V)
    r_s = panel.loc[S["fila"], "ratio_t1"].values
    cache = {}
    def p_c(c):
        if c not in cache:
            mod = M.construir(espec, params); mod.set_params(n_jobs=8); mod.fit(Xs, (r_s < c).astype(int))
            cache[c] = mod.predict_proba(Xv)[:,1]
        return cache[c]
    # efecto de corredor (offset en log-odds): TE cruzado del corredor, contrastado contra la media global
    ks = (S["mercado"]+"|"+S["cod_mpio"]).to_frame("c"); kv = (V["mercado"]+"|"+V["cod_mpio"]).to_frame("c")
    te = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=42).fit(ks, ys)
    off_v = logit(np.clip(te.transform(kv).ravel(),1e-3,1-1e-3)) - logit(ys.mean())
    fila = {"pliegue": pl, "XGB": roc_auc_score(yv, p_c(0.5))}
    for g, cs in grids.items():
        s = np.mean([p_c(c) for c in cs], axis=0); fila[g] = roc_auc_score(yv, s)
        fila[g+"_logit"] = roc_auc_score(yv, np.mean([logit(np.clip(p_c(c),1e-6,1-1e-6)) for c in cs], axis=0))
    for lam in [0.1, 0.2, 0.3]:
        s = np.mean([p_c(c) for c in grids["g5"]], axis=0)
        fila[f"g5+corr{lam}"] = roc_auc_score(yv, logit(np.clip(s,1e-6,1-1e-6)) + lam*off_v)
    res.append(fila); print({k: (round(v,4) if isinstance(v,float) else v) for k,v in fila.items()}, flush=True)
r = pd.DataFrame(res).set_index("pliegue"); print((r.sub(r["XGB"], axis=0)).mean().round(4)); print(r.mean().round(4))
