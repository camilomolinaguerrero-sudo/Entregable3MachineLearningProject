import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, f1_score
from sklearn.preprocessing import TargetEncoder
import experimento as X, modelos as M, preprocesamiento as P, config as C, datos as D, evaluacion as EV
m = pd.read_parquet(sys.argv[2])
panel = D.cargar_panel().set_index("fila")
ext = m[(m.tarea=="clf")&(m.combinacion=="XGBoost | Sin balanceo | Random Search")]
res = []
for pl in ["k1","k2","k3","k4"]:
    S, V = X.cargar_pliegue(pl)
    params = json.loads(ext[ext.pliegue==pl].params_json.iloc[0])
    espec = M.catalogo("clf")["XGBoost"]
    prep = P.preprocesador("arboles","clf")
    ys = S[C.OBJ_CLF].values; yv = V[C.OBJ_CLF].values
    Xs = prep.fit_transform(S, ys); Xv = prep.transform(V)
    r_s = panel.loc[S["fila"], "ratio_t1"].values
    ks = (S["mercado"]+"|"+S["cod_mpio"]).to_frame("c"); kv = (V["mercado"]+"|"+V["cod_mpio"]).to_frame("c")
    te = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=42)
    Xs2 = np.c_[Xs, te.fit_transform(ks, ys)]; Xv2 = np.c_[Xv, te.transform(kv)]
    def fp(Xa, Xb, y):
        mod = M.construir(espec, params); mod.set_params(n_jobs=8); mod.fit(Xa, y); return mod.predict_proba(Xb)[:,1]
    cs = [0.3,0.4,0.5,0.6,0.7]
    out = {"XGB": fp(Xs, Xv, ys), "umbrales": np.mean([fp(Xs, Xv, (r_s<c).astype(int)) for c in cs], axis=0),
           "XGB+corr": fp(Xs2, Xv2, ys), "umbrales+corr": np.mean([fp(Xs2, Xv2, (r_s<c).astype(int)) for c in cs], axis=0)}
    fila = {"pliegue": pl}
    for k, s in out.items():
        fila[k] = roc_auc_score(yv, s); fila[k+" PR"] = average_precision_score(yv, s); fila[k+" Brier"] = brier_score_loss(yv, s); fila[k+" LL"] = log_loss(yv, s); fila[k+" ECE"] = EV.ece(yv, s)
    res.append(fila); print(pl, {k: round(v,4) for k,v in fila.items() if k!="pliegue" and (" " not in k)}, flush=True)
r = pd.DataFrame(res).set_index("pliegue"); print(r.mean().round(5).to_string())
