import sys, json, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
import experimento as X, modelos as M, preprocesamiento as P, config as C
m = pd.read_parquet(sys.argv[2])
ext = m[(m.tarea=="clf")&(m.combinacion=="XGBoost | Sin balanceo | Random Search")]
for pl in ["k1","k2","k3","k4"]:
    S, V = X.cargar_pliegue(pl)
    params = json.loads(ext[ext.pliegue==pl].params_json.iloc[0])
    espec = M.catalogo("clf")["XGBoost"]
    prep = P.preprocesador("arboles","clf")
    Xs = prep.fit_transform(S, S[C.OBJ_CLF].values); Xv = prep.transform(V)
    mod = M.construir(espec, params); mod.set_params(n_jobs=8); mod.fit(Xs, S[C.OBJ_CLF].values)
    a0 = roc_auc_score(V[C.OBJ_CLF], mod.predict_proba(Xv)[:,1])
    # corredor = mercado|cod_mpio con TE cruzado
    ks = (S["mercado"]+"|"+S["cod_mpio"]).to_frame("c"); kv = (V["mercado"]+"|"+V["cod_mpio"]).to_frame("c")
    te = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=42)
    ts = te.fit_transform(ks, S[C.OBJ_CLF].values); tv = te.transform(kv)
    mod2 = M.construir(espec, params); mod2.set_params(n_jobs=8); mod2.fit(np.c_[Xs, ts], S[C.OBJ_CLF].values)
    a1 = roc_auc_score(V[C.OBJ_CLF], mod2.predict_proba(np.c_[Xv, tv])[:,1])
    # cobertura: fracción de filas de V cuyo corredor aparece en S y conteo medio
    cnt = ks["c"].value_counts(); cov = kv["c"].isin(cnt.index).mean(); med = kv["c"].map(cnt).fillna(0).median()
    print(pl, "XGB", round(a0,4), "XGB+TE corredor", round(a1,4), "Δ", round(a1-a0,4), "| cobertura", round(cov,3), "filas S por corredor (mediana en V)", med, flush=True)
