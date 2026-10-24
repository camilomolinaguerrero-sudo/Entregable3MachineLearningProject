import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import experimento as X, modelos as M, preprocesamiento as P, config as C, datos as D
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
    def fit(y, Xtr=Xs, w=None):
        mod = M.construir(espec, params); mod.set_params(n_jobs=8); mod.fit(Xtr, y, sample_weight=w); return mod
    base = fit(ys).predict_proba(Xv)[:,1]
    # (a) valla: P(cero) + P(no cero) * P(ratio<0.5 | no cero)
    z = (r_s == 0).astype(int)
    pz = fit(z).predict_proba(Xv)[:,1]
    nz = r_s > 0
    pc = fit((r_s[nz] < 0.5).astype(int), Xs[nz]).predict_proba(Xv)[:,1]
    valla = pz + (1-pz)*pc
    # (b) umbrales acumulados: promedio de P(ratio < c)
    cs = [0.3, 0.4, 0.5, 0.6, 0.7]
    acum = np.mean([fit((r_s < c).astype(int)).predict_proba(Xv)[:,1] for c in cs], axis=0)
    fila = {"pliegue": pl}
    for nombre, s in [("XGB", base), ("valla", valla), ("acumulados", acum), ("valla+base", (valla+base)/2)]:
        fila[nombre] = roc_auc_score(yv, s); fila[nombre+"_pr"] = average_precision_score(yv, s)
    res.append(fila); print({k: (round(v,4) if isinstance(v,float) else v) for k,v in fila.items()}, flush=True)
r = pd.DataFrame(res); print(r.mean(numeric_only=True).round(4))
