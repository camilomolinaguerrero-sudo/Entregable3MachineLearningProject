import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import experimento as X, modelos as M, preprocesamiento as P, config as C, datos as D
m = pd.read_parquet(sys.argv[2])
panel = D.cargar_panel().set_index("fila")
ext = m[(m.tarea=="clf")&(m.combinacion=="XGBoost | Sin balanceo | Random Search")]
res = []
def pava_filas(Pm):
    # proyección isotónica fila por fila (no decreciente en el umbral) con PAVA vectorizado simple
    out = Pm.copy()
    for i in range(out.shape[1]-1):
        pass
    return np.maximum.accumulate(out, axis=1)  # aproximación rápida (no es la proyección L2 exacta)
for pl in ["k1","k2","k3","k4"]:
    S, V = X.cargar_pliegue(pl)
    params = json.loads(ext[ext.pliegue==pl].params_json.iloc[0])
    espec = M.catalogo("clf")["XGBoost"]
    prep = P.preprocesador("arboles","clf")
    ys = S[C.OBJ_CLF].values; yv = V[C.OBJ_CLF].values
    Xs = prep.fit_transform(S, ys); Xv = prep.transform(V)
    r_s = panel.loc[S["fila"], "ratio_t1"].values
    def fit_pred(y, seed=42):
        mod = M.construir(espec, params, semilla=seed); mod.set_params(n_jobs=8); mod.fit(Xs, y); return mod.predict_proba(Xv)[:,1]
    base = fit_pred(ys)
    semillas = np.mean([fit_pred(ys, 42+i) for i in range(5)], axis=0)
    cs = [0.3,0.4,0.5,0.6,0.7]
    Pm = np.column_stack([fit_pred((r_s < c).astype(int)) for c in cs])
    umb = Pm.mean(axis=1)
    umb_iso = pava_filas(Pm).mean(axis=1)
    cs9 = np.linspace(0.35, 0.65, 9)
    P9 = np.column_stack([fit_pred((r_s < c).astype(int)) for c in cs9])
    umb9 = P9.mean(axis=1)
    # umbrales y semillas a la vez: cada umbral con semilla distinta
    Pms = np.column_stack([fit_pred((r_s < c).astype(int), 42+i) for i, c in enumerate(cs)]).mean(axis=1)
    fila = {"pliegue": pl}
    for n, s in [("XGB", base), ("5 semillas (c=0,5)", semillas), ("5 umbrales", umb), ("5 umbrales iso", umb_iso), ("9 umbrales estrechos", umb9), ("5 umbrales×semillas", Pms)]:
        fila[n] = roc_auc_score(yv, s); fila[n+" PR"] = average_precision_score(yv, s); fila[n+" Brier"] = brier_score_loss(yv, s)
    res.append(fila); print(pl, {k: round(v,4) for k,v in fila.items() if k!="pliegue" and "Brier" not in k}, flush=True)
r = pd.DataFrame(res).set_index("pliegue")
print(r.mean().round(4).to_string())
