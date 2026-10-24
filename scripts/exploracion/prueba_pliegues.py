import sys, json, warnings
warnings.filterwarnings("ignore")
RAIZ = sys.argv[1]; pl = sys.argv[2]
sys.path.insert(0, RAIZ + "/src")
import importlib.util, joblib, numpy as np, pandas as pd
import config as C, datos as D, experimento as X, modelos as M, preprocesamiento as P, evaluacion as EV
from ejul import EnsambleUmbralesLatentes
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
spec = importlib.util.spec_from_file_location("m12", RAIZ + "/scripts/12_modelo_original.py"); m12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m12)
out = joblib.load(f"{RAIZ}/data/resultados/original/{pl}.joblib"); res = out["resumen"]
panel = D.cargar_panel(); G = m12.codigos_grupo(panel)
razon = pd.Series(panel["ratio_t1"].to_numpy(), index=panel["fila"].to_numpy())
S, V = X.cargar_pliegue(pl)
prep = P.preprocesador("arboles", "clf")
Xs, Xv = m12.matriz(prep, S, G, True), m12.matriz(prep, V, G)
ys, yv = S[C.OBJ_CLF].to_numpy(), V[C.OBJ_CLF].to_numpy()
base = M.construir(M.catalogo("clf")["XGBoost"], res["params_base"]); base.set_params(n_jobs=2)
mod = EnsambleUmbralesLatentes(estimador_base=base, columnas_grupo=(-3, -2, -1), random_state=C.SEMILLA, prediccion="pliegues", **res["params"])
mod.fit(Xs, ys, latente=razon.loc[S["fila"]].to_numpy())
p_new = mod.predict_proba(Xv)[:, 1]
p_old = pd.Series(out["pred"]["s"], index=out["pred"]["fila"]).loc[V["fila"]].to_numpy()
pe = pd.read_parquet(RAIZ + "/data/resultados/pred_externo_clf.parquet") if pl != "final" else pd.read_parquet(RAIZ + "/data/resultados/pred_prueba_clf.parquet")
px = pe.set_index("fila").loc[V["fila"], "XGBoost | Sin balanceo | Random Search"].to_numpy()
def met(p): return {"auc": roc_auc_score(yv, p), "pr": average_precision_score(yv, p), "brier": brier_score_loss(yv, p), "ll": log_loss(yv, p), "ece": EV.ece(yv, p)}
r = pd.DataFrame({"xgb": met(px), "completo": met(p_old), "pliegues": met(p_new)}).T
print(pl); print(r.round(5).to_string(), flush=True)
r.assign(pliegue=pl).to_csv(f"/private/tmp/claude-501/-Users-cjmolina-Documents-mis-finanzas/f5b187a6-d8f7-417f-9e1c-068324c76e7a/scratchpad/e3/pliegues_{pl}.csv")
