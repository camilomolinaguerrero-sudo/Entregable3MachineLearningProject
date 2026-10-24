import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
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
    edad = (S["semana_t1"].max() - S["semana_t1"]).dt.days.values / 365.25
    fila = {"pliegue": pl}
    for lam in [0.0, 0.25, 0.5, 1.0]:
        w = np.exp(-lam * edad)
        mod = M.construir(espec, params); mod.set_params(n_jobs=8); mod.fit(Xs, ys, sample_weight=w)
        s = mod.predict_proba(Xv)[:,1]
        fila[f"λ={lam}"] = roc_auc_score(yv, s)
        if lam == 0.5:
            Pm = []
            for c in [0.3,0.4,0.5,0.6,0.7]:
                mm = M.construir(espec, params); mm.set_params(n_jobs=8); mm.fit(Xs, (r_s<c).astype(int), sample_weight=w); Pm.append(mm.predict_proba(Xv)[:,1])
            fila["λ=0.5 + umbrales"] = roc_auc_score(yv, np.mean(Pm, axis=0))
    res.append(fila); print({k: (round(v,4) if isinstance(v,float) else v) for k,v in fila.items()}, flush=True)
r = pd.DataFrame(res).set_index("pliegue"); print(r.mean().round(4).to_string()); print((r.sub(r["λ=0.0"],axis=0)).mean().round(4).to_string())
