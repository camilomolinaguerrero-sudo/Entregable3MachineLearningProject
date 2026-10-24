import sys, json, warnings, time
warnings.filterwarnings("ignore")
RAIZ = sys.argv[1]; pl = sys.argv[2]; nj = int(sys.argv[3])
sys.path.insert(0, RAIZ + "/src")
import importlib.util, joblib, numpy as np, pandas as pd
import config as C, datos as D, experimento as X, modelos as M, preprocesamiento as P, evaluacion as EV
from ejul import EnsambleUmbralesLatentes
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
spec = importlib.util.spec_from_file_location("m12", RAIZ + "/scripts/12_modelo_original.py"); m12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m12)
res = joblib.load(f"{RAIZ}/data/resultados/original/{pl}.joblib")["resumen"]
panel = D.cargar_panel(); G = m12.codigos_grupo(panel)
S, V = X.cargar_pliegue(pl)
full = panel[(panel.semana_t1 >= S.semana_t1.min()) & (panel.semana_t1 <= S.semana_t1.max())]
assert full.semana_t1.max() < V.semana_t1.min()
print(pl, "filas completas", len(full), flush=True)
prep = P.preprocesador("arboles", "clf")
Xf, Xv = m12.matriz(prep, full, G, True), m12.matriz(prep, V, G)
yf, yv = full[C.OBJ_CLF].to_numpy(), V[C.OBJ_CLF].to_numpy()
def met(p): return {"auc": roc_auc_score(yv, p), "pr": average_precision_score(yv, p), "brier": brier_score_loss(yv, p), "ll": log_loss(yv, p), "ece": EV.ece(yv, p)}
def base():
    b = M.construir(M.catalogo("clf")["XGBoost"], res["params_base"]); b.set_params(n_jobs=nj); return b
out = {}
t0 = time.time(); out["xgb"] = met(base().fit(Xf[:, :-3], yf).predict_proba(Xv[:, :-3])[:, 1]); print("xgb", round(time.time()-t0), out["xgb"], flush=True)
params = {k: v for k, v in res["params"].items()}
for cod in ["temporal", "cruzada"]:
    t0 = time.time()
    m = EnsambleUmbralesLatentes(estimador_base=base(), columnas_grupo=(-3, -2, -1), random_state=C.SEMILLA, codificacion=cod, **params)
    m.fit(Xf, yf, latente=full["ratio_t1"].to_numpy(), tiempos=full["semana_t1"].to_numpy())
    out[cod] = met(m.predict_proba(Xv)[:, 1]); print(cod, round(time.time()-t0), out[cod], flush=True)
pd.DataFrame(out).T.to_csv(f"/private/tmp/claude-501/-Users-cjmolina-Documents-mis-finanzas/f5b187a6-d8f7-417f-9e1c-068324c76e7a/scratchpad/e3/escala_{pl}.csv")
print(pd.DataFrame(out).T.round(5).to_string())
