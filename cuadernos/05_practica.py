# %% [markdown]
# ## 5.6 Implementación
#
# La clase `ApilamientoJerarquicoKL` (`src/original_model.py`) hereda de `BaseEstimator` y `ClassifierMixin`,
# así que funciona con `clone`, `Pipeline` y la validación anidada. Toda la lógica propia usa solo numpy y
# scipy: el objetivo y el gradiente de cada nodo, la actualización multiplicativa cerrada, la brecha de
# Frank-Wolfe, el recorrido de la jerarquía y la calibración beta por Newton-Raphson ponderado. No hay
# ningún `import` de un combinador ya hecho. La entrada `X` tiene $M$ columnas de probabilidades fuera de
# muestra de los modelos base seguidas de los códigos enteros de la jerarquía (central, central × grupo,
# corredor). En producción, la clase `AJKLProduccion` (`src/artefactos.py`) arma esa matriz a partir de las 34
# variables crudas: calcula las probabilidades con los 9 artefactos base y traduce los códigos con los
# diccionarios guardados en el entrenamiento.

# %%
import inspect
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import Markdown, display

sys.path.insert(0, "../src")
sys.path.insert(0, "../scripts")
warnings.filterwarnings("ignore")
pio.renderers.default = "notebook_connected"
import original_model as OM
import plots as PL

R = "../data/resultados/"
PL_ORDEN = ["k1", "k2", "k3", "k4", "final"]
for f in [OM.brecha_fw, OM._paso_espejo, OM.descenso_espejo]:
    print(inspect.getsource(f))

# %% [markdown]
# ### Pruebas unitarias
#
# `tests/test_original_model.py` tiene 20 pruebas. Cubren los cuatro mínimos de la guía y las propiedades
# del capítulo: (i) aprende en datos sintéticos con solución conocida (dos regiones donde cada modelo base es
# el bueno en una; el AJ-KL recupera pesos > 0,8 para el modelo correcto en cada región y supera al apilamiento
# global en más de 0,03 de AUC); (ii) es reproducible (el algoritmo es determinista: no tiene semilla porque
# no tiene ningún paso aleatorio); (iii) rechaza entradas e hiperparámetros inválidos; (iv) es compatible con
# `clone`. Además: el descenso espejo alcanza el óptimo de SLSQP y, con nueve modelos casi colineales y los
# valores por defecto, el de L-BFGS con brecha de Frank-Wolfe < 10⁻⁵; los casos límite de λ (proposición 4);
# la KL; el olvido; el respaldo al ancestro para nodos no vistos; la calibración beta; y el Newton ponderado
# frente a scikit-learn.

# %%
import subprocess
r = subprocess.run([sys.executable, "-m", "pytest", "../tests/test_original_model.py", "-q", "--no-header",
                    "-p", "no:cacheprovider"], capture_output=True, text=True)
print(r.stdout[-600:])

# %% [markdown]
# ### Hiperparámetros elegidos en cada pliegue
#
# TPE (Optuna, semilla 42, 24 evaluaciones, 8 de arranque) ajustó λ ∈ [0,01; 1000] (escala log), γ ∈ [0,98; 1]
# y la profundidad ∈ {1, 2, 3, 4}, con el AUC de la validación interna como criterio, igual que para los
# benchmarks. La validación interna es hacia adelante dentro del historial del combinador: se entrena con las
# semanas anteriores y se evalúa en las últimas 26 semanas (en k1, entre las ventanas internas de S_k1). La
# calibración beta se activa solo si baja la log-loss en esa misma validación.

# %%
res = {pl: joblib.load(f"{R}ajkl/{pl}.joblib") for pl in PL_ORDEN}
tab_hp = pd.DataFrame({pl: {**r["resumen"]["params"], "filas del historial": r["resumen"]["filas_historial"],
                            "historial": r["resumen"]["historial_bloques"], "nodos por nivel": r["resumen"]["nodos"],
                            "segundos (búsqueda + reajuste)": round(r["resumen"]["t_total"])}
                       for pl, r in res.items()}).T
display(tab_hp)
hist = pd.concat([r["historial"].assign(pliegue=pl) for pl, r in res.items()])
PL.convergencia_tpe(hist).show()

# %% [markdown]
# ## 5.7 Convergencia y complejidad empírica
#
# Para el combinador del pliegue final se calcula, en cada nodo, la brecha de Frank-Wolfe de los pesos
# ajustados, que acota el error de optimización (proposición 3a), y el número de iteraciones aceptadas.

# %%
import importlib.util
spec = importlib.util.spec_from_file_location("m22", "../scripts/22_ajkl.py")
m22 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m22)
import datos as D
sel = m22.seleccion()
cols = m22.columnas_base(sel)
G = m22.codigos_jerarquia(D.cargar_panel())
h, _ = m22.historial("final", sel)
Xh, yh, th = m22.matriz(h, cols, G), h["interrupcion_t1"].to_numpy(float), m22.semanas(h["semana_t1"])
mod = res["final"]["modelo"]
P = Xh[:, :9]
a = mod.gamma ** (th.max() - th)
filas = [{"nivel": "raíz", "nodo": "raíz", "n": len(yh), "iteraciones": len(mod.historial_[("raiz",)]) - 1,
          "brecha": OM.brecha_fw(mod.raiz_, P, yh, a, np.full(9, 1 / 9), mod.lam / a.sum())}]
nombres_nivel = ["central", "central × grupo", "corredor"]
for l, tabla in enumerate(mod.pesos_):
    cod = Xh[:, 9 + l].astype(int)
    orden = np.argsort(cod, kind="stable")
    unicos, inicios = np.unique(cod[orden], return_index=True)
    fines = np.r_[inicios[1:], len(orden)]
    grupos = {int(c): orden[i:f] for c, i, f in zip(unicos, inicios, fines)}
    for c, w in tabla.items():
        idx = grupos[c]
        padre = mod.raiz_ if l == 0 else mod.pesos_[l - 1].get(int(Xh[idx[0], 9 + l - 1]), mod.raiz_)
        filas.append({"nivel": nombres_nivel[l], "nodo": c, "n": len(idx), "iteraciones": len(mod.historial_[(l, c)]) - 1,
                      "brecha": OM.brecha_fw(w, P[idx], yh[idx], a[idx], padre, mod.lam / a[idx].sum())})
conv = pd.DataFrame(filas)
display(conv.groupby("nivel", sort=False).agg(nodos=("nodo", "size"), filas_mediana=("n", "median"),
                                              iteraciones_mediana=("iteraciones", "median"), iteraciones_max=("iteraciones", "max"),
                                              brecha_mediana=("brecha", "median"), brecha_max=("brecha", "max")))
fig = go.Figure()
hr = mod.historial_[("raiz",)]
fig.add_scatter(y=hr - hr.min() + 1e-12, mode="lines", line=dict(color=PL.NARANJA), name="raíz")
fig.update_layout(title="Raíz del pliegue final: J(w_k) − J(w_final) por iteración aceptada", height=320)
fig.update_yaxes(type="log", title="Exceso de objetivo")
fig.update_xaxes(title="Iteración")
fig.show()

# %% [markdown]
# **Tiempos frente a n.** Se reajusta el combinador del pliegue final con submuestras del historial
# (mediana de 3 repeticiones) y se mide la predicción de 1.000 filas. Los tiempos no incluyen los modelos
# base, que ya están entrenados; su costo está en el capítulo 3.

# %%
filas_t = []
rng = np.random.default_rng(0)
for n in [10_000, 30_000, 100_000, len(yh)]:
    idx = np.sort(rng.choice(len(yh), min(n, len(yh)), replace=False))
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        m = OM.ApilamientoJerarquicoKL(**mod.get_params()).fit(Xh[idx], yh[idx], tiempos=th[idx])
        ts.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    m.predict_proba(Xh[:1000])
    filas_t.append({"n": len(idx), "segundos_fit": float(np.median(ts)), "ms_por_1000_pred": 1000 * (time.perf_counter() - t0)})
tt = pd.DataFrame(filas_t)
pend = np.polyfit(np.log(tt["n"]), np.log(tt["segundos_fit"]), 1)[0]
display(tt.round(3))
display(Markdown(f"Pendiente log-log del tiempo de entrenamiento frente a n: **{pend:.2f}** (1 = lineal)."))

# %% [markdown]
# ## 5.8 Ablación y reglas de combinación
#
# Cada fila cambia un componente del modelo elegido en cada pliegue, o lo reemplaza por una regla de
# combinación estándar ajustada con **el mismo historial**, y reporta la diferencia media frente al AJ-KL
# completo en los cinco pliegues externos (negativo en AUC y AUC-PR, o positivo en Brier, log-loss y ECE,
# significa que la variante es peor). Las reglas estándar son el promedio simple, los pesos por inverso de la
# log-loss, el stacking logístico (regresión logística sobre los logit de los nueve modelos) y el pool lineal
# transformado con beta (BLP) de {cite:t}`ranjan2010combining`. Dos controles responden a la revisión del
# diseño: el AJ-KL con solo los 7 modelos del curso como base (¿la mejora viene de incluir LightGBM y
# CatBoost?) y con el historial reducido a 40 mil filas, el tamaño de la muestra $S_k$ de los benchmarks (¿la
# mejora viene de ver más filas etiquetadas?). Las variantes reutilizan los hiperparámetros del modelo
# completo, sin reajuste.

# %%
met = ["auc", "auc_pr", "brier", "log_loss", "ece"]
cmp_ = pd.concat([r["comparacion"].assign(pliegue=pl) for pl, r in res.items()])
base = cmp_[cmp_.variante == "AJ-KL (completo)"].set_index("pliegue")[met]
dif = cmp_.set_index(["variante", "pliegue"])[met] - base
tabla_abl = dif.groupby(level=0).mean()
tabla_abl["pliegues en que pierde AUC"] = (dif["auc"] < 0).groupby(level=0).sum()
tabla_abl["pliegues"] = dif["auc"].groupby(level=0).size()
display(tabla_abl.drop("AJ-KL (completo)").sort_values("auc").round(5))

# %% [markdown]
# ## 5.9 Iteraciones del diseño
#
# El modelo reportado es la versión 2.1. Las iteraciones se reportan completas porque varias se decidieron
# después de ver resultados de los bloques externos, y el lector debe poder juzgar cuánto pesa esa información
# (riesgo de «caminos que se bifurcan»).
#
# | Versión | Cambio | Evidencia que lo motivó | Resultado |
# |---|---|---|---|
# | EJUL (exploratorio) | Ensamble de umbrales latentes con codificación Bayes empírica sobre XGBoost | Primera propuesta, anterior a la revisión de literatura del equipo | ΔAUC externo +0,0010 frente a XGBoost, ECE peor; se conserva como anexo |
# | AJ-KL v1 | Propuesta de la revisión: 7 modelos base, jerarquía de 3 niveles, combinador entrenado con las predicciones internas de $S_k$ | Revisión de literatura (capítulo 4) | No superó a XGBoost: las predicciones internas vienen de modelos entrenados con menos datos que los que se combinan |
# | AJ-KL v2.0 | 9 modelos base (con LightGBM y CatBoost), nivel de corredor, historial con los bloques externos anteriores | Diagnóstico de v1: el combinador debe aprender con predicciones del mismo régimen | Superó a XGBoost en los bloques externos por márgenes pequeños |
# | AJ-KL v2.1 (final) | Optimizador resuelto hasta la brecha de Frank-Wolfe (con pasos de Frank-Wolfe cuando el paso multiplicativo se estanca); η0 y K fuera del TPE; historial sin las 8 semanas del hueco | Revisión adversarial del código: la parada en K ≤ 300 dejaba los pesos lejos del óptimo, y el historial de k2-k4 incluía semanas del hueco que los benchmarks no ven | Resultados de este capítulo y del 6 |
#
# Los resultados de 2025 de las versiones anteriores se conocían cuando se hizo cada cambio. Los dos cambios
# de la versión 2.1 no se decidieron por desempeño sino por corrección: el primero hace que el estimador sea el
# de la formulación y el segundo iguala la información disponible con la de los benchmarks. Los resultados de
# v1 y v2.0 están en `data/resultados/iteraciones/`.

# %%
v20 = {}
for pl in PL_ORDEN:
    try:
        v20[pl] = joblib.load(f"{R}iteraciones/ajkl_v2_0/{pl}.joblib")["resumen"]["auc"]
    except FileNotFoundError:
        pass
it = pd.DataFrame({"AUC v2.0": pd.Series(v20), "AUC v2.1": pd.Series({pl: r["resumen"]["auc"] for pl, r in res.items()}),
                   "AUC XGBoost": cmp_[cmp_.variante == "Modelo base: XGBoost"].set_index("pliegue")["auc"]})
display(it.round(4))

# %% [markdown]
# ## 5.10 Interpretabilidad
#
# El AJ-KL es interpretable por construcción: la predicción de una fila es un promedio ponderado de nueve
# probabilidades y los pesos dependen solo del nodo. La primera figura muestra los pesos de la raíz por
# pliegue; la segunda, los pesos por central del pliegue final. El tablero muestra, para cada predicción, el
# aporte $w_m p_m$ de cada modelo base y los valores SHAP sobre las 34 variables crudas.

# %%
nb = [c.split(" | ")[0] for c in cols]
raices = pd.DataFrame({pl: r["modelo"].raiz_ for pl, r in res.items()}, index=nb).T
fig = go.Figure()
for c in nb:
    fig.add_bar(x=raices.index, y=raices[c], name=c)
fig.update_layout(barmode="stack", title="Pesos de la raíz por pliegue (suman 1)", height=380, yaxis_title="Peso")
fig.show()
centrales = pd.factorize(D.cargar_panel()["mercado"].astype(str))[1]
if mod.pesos_:
    pc = pd.DataFrame([{"central": centrales[k], **dict(zip(nb, w))} for k, w in mod.pesos_[0].items()]).set_index("central")
    fig = go.Figure(go.Heatmap(z=pc.to_numpy(), x=nb, y=pc.index, colorscale="Blues", colorbar=dict(title="Peso")))
    fig.update_layout(title="Pesos por central mayorista (pliegue final)", height=760)
    fig.show()
    ent = -(pc * np.log(pc.clip(1e-12))).sum(axis=1)
    display(Markdown("Modelo de mayor peso por central: " + ", ".join(f"{k} ({v})" for k, v in pc.idxmax(axis=1).value_counts().items())
                     + f". Entropía media de los pesos: {ent.mean():.2f} nats (máximo posible {np.log(9):.2f})."))

# %%
sg = pd.read_parquet("../data/dashboard/shap_global.parquet")
if "ajkl_original" in set(sg["modelo"]):
    piv = sg.pivot(index="variable", columns="modelo", values="importancia")
    top = piv.sort_values("ajkl_original", ascending=False).head(15)[["ajkl_original", "xgboost"]]
    fig = go.Figure()
    fig.add_bar(y=top.index[::-1], x=top["ajkl_original"][::-1], orientation="h", name="AJ-KL", marker_color=PL.NARANJA)
    fig.add_bar(y=top.index[::-1], x=top["xgboost"][::-1], orientation="h", name="XGBoost", marker_color=PL.AZUL)
    fig.update_layout(title="Importancia global: media de |SHAP| (puntos de probabilidad), 15 variables principales",
                      barmode="group", height=560)
    fig.show()
    from scipy.stats import spearmanr
    display(Markdown(f"Correlación de Spearman entre las importancias del AJ-KL y de XGBoost: "
                     f"{spearmanr(piv['ajkl_original'], piv['xgboost']).correlation:.2f}."))

# %% [markdown]
# ## 5.11 Generalización a otros conjuntos de datos
#
# La guía pide al menos tres conjuntos adicionales. Se usaron tres conjuntos públicos de la UCI con orden
# temporal y grupos anidados, de naturaleza distinta al SIPSA: (1) **Beijing PM2.5** (12 estaciones,
# 2013-2017): ¿superará el PM2.5 del día siguiente 75 µg/m³?, jerarquía estación → estación × estación del año
# → estación × mes; (2) **Bike Sharing** (Washington D. C., 2011-2012, por hora): ¿caerá la demanda de la hora
# siguiente por debajo de la mitad de su media de la misma hora en los 7 días anteriores?, jerarquía franja →
# hora → hora × día laboral; (3) **Bank Marketing** (Portugal, 2008-2010): ¿el cliente suscribe el depósito?,
# sin la variable `duration` (que se conoce solo después de la llamada), jerarquía ocupación → ocupación ×
# educación → ocupación × educación × estado civil. Cada conjunto se dividió en 6 tramos temporales con 5
# pliegues de origen creciente. Los 9 modelos base se ajustaron con el mismo protocolo (TPE con 24 evaluaciones
# sobre ventanas internas temporales) y el combinador de cada pliegue se entrenó con las predicciones de los
# tramos anteriores; por eso la comparación del AJ-KL empieza en el pliegue 2 (4 pliegues por conjunto).

# %%
ext = pd.read_parquet(R + "externos/comparacion_v21.parquet")
mods = ["AJ-KL", "AJ-KL global (profundidad 1)", "XGBoost", "LightGBM", "CatBoost", "Random Forest", "Regresión logística", "SVM",
        "KNN", "Decision Tree", "Naive Bayes"]
display(ext.groupby("nombre")[mods].mean().T.round(4))
filas_e = []
for nombre, g in ext.groupby("nombre"):
    bases = mods[2:]
    mejor_base = g[bases].mean().idxmax()
    filas_e.append({"conjunto": nombre, "AUC AJ-KL": g["AJ-KL"].mean(), "mejor modelo base": mejor_base,
                    "AUC mejor base": g[mejor_base].mean(), "pliegues AJ-KL > XGBoost": int((g["AJ-KL"] > g["XGBoost"]).sum()),
                    "pliegues AJ-KL > mejor base": int((g["AJ-KL"] > g[mejor_base]).sum()), "pliegues": len(g),
                    "Δ log-loss vs XGBoost": (g["AJ-KL | log_loss"] - g["XGBoost | log_loss"]).mean()})
display(pd.DataFrame(filas_e).round(4))

# %% [markdown]
# ## 5.12 Dónde el AJ-KL no supera a XGBoost
#
# Desempeño en la prueba 2025 por subgrupos, con las predicciones del pliegue final de la validación anidada.

# %%
panel = D.cargar_panel().set_index("fila")
pru = pd.read_parquet(R + "pred_prueba_clf.parquet")
p = res["final"]["pred"]
pru["AJ-KL"] = pd.Series(p["s"], index=p["fila"]).reindex(pru["fila"]).to_numpy()
pru["XGBoost"] = pru["XGBoost | Sin balanceo | Random Search"]
info = panel.loc[pru["fila"], ["dist_km", "pdet_origen", "mercado", "log_base8"]].reset_index(drop=True)
pru = pd.concat([pru.reset_index(drop=True), info], axis=1)
pru["distancia"] = pd.qcut(pru["dist_km"], 4, labels=["Q1 corta", "Q2", "Q3", "Q4 larga"])
pru["escala"] = pd.qcut(pru["log_base8"], 4, labels=["Q1 pequeño", "Q2", "Q3", "Q4 grande"])
pru["PDET"] = pru["pdet_origen"].map({0: "No PDET", 1: "PDET"})
from sklearn.metrics import roc_auc_score
filas = []
for grupo in ["distancia", "escala", "PDET"]:
    for nivel, g in pru.groupby(grupo, observed=True):
        filas.append({"grupo": grupo, "nivel": nivel, "n": len(g), "tasa": g["interrupcion_t1"].mean(),
                      "AUC AJ-KL": roc_auc_score(g["interrupcion_t1"], g["AJ-KL"]),
                      "AUC XGBoost": roc_auc_score(g["interrupcion_t1"], g["XGBoost"])})
sub = pd.DataFrame(filas)
sub["ΔAUC"] = sub["AUC AJ-KL"] - sub["AUC XGBoost"]
display(sub.round(4))
cent = pru.groupby("mercado").apply(lambda g: roc_auc_score(g["interrupcion_t1"], g["AJ-KL"]) - roc_auc_score(g["interrupcion_t1"], g["XGBoost"])
                                    if g["interrupcion_t1"].nunique() > 1 else np.nan).dropna()
display(Markdown(f"Por central mayorista, el AJ-KL tiene mayor AUC que XGBoost en {(cent > 0).sum()} de {len(cent)} centrales; "
                 f"la mayor desventaja es {cent.min():+.4f} ({cent.idxmin()}) y la mayor ventaja {cent.max():+.4f} ({cent.idxmax()})."))
