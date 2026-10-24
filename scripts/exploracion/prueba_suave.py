import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from scipy.special import expit
from xgboost import XGBRegressor
import experimento as X, modelos as M, preprocesamiento as P, config as C, datos as D
m = pd.read_parquet(sys.argv[2])
panel = D.cargar_panel().set_index("fila")
ext = m[(m.tarea=="clf")&(m.combinacion=="XGBoost | Sin balanceo | Random Search")]
res = []
for pl in ["k1","k2","k3","k4"]:
    S, V = X.cargar_pliegue(pl)
    params = json.loads(ext[ext.pliegue==pl].params_json.iloc[0])
    prep = P.preprocesador("arboles","clf")
    ys = S[C.OBJ_CLF].values; yv = V[C.OBJ_CLF].values
    Xs = prep.fit_transform(S, ys); Xv = prep.transform(V)
    r = panel.loc[S["fila"], "ratio_t1"].values
    def fit(lbl):
        mod = XGBRegressor(objective="reg:logistic", tree_method="hist", n_estimators=300, n_jobs=8, random_state=42, verbosity=0, **params)
        mod.fit(Xs, lbl); return mod.predict(Xv)
    fila = {"pliegue": pl, "duro": None}
    s0 = fit(ys.astype(float)); fila["duro"] = roc_auc_score(yv, s0); fila["duro_ll"] = log_loss(yv, s0)
    for h in [0.1, 0.2, 0.3]:
        lbl = np.clip((0.5 + h - r) / (2*h), 0, 1); s = fit(lbl)
        fila[f"rampa h={h}"] = roc_auc_score(yv, s); fila[f"rampa h={h}_pr"] = average_precision_score(yv, s); fila[f"rampa h={h}_ll"] = log_loss(yv, s)
    for sc in [0.05, 0.1]:
        lbl = expit((0.5 - r)/sc); s = fit(lbl)
        fila[f"logística s={sc}"] = roc_auc_score(yv, s); fila[f"logística s={sc}_ll"] = log_loss(yv, s)
    res.append(fila); print({k: (round(v,4) if isinstance(v,float) else v) for k,v in fila.items()}, flush=True)
rr = pd.DataFrame(res).set_index("pliegue"); print(rr.mean().round(4).to_string())
