# %% [markdown]
# # 6. Validación estadística
#
# El modelo original se considera superior solo si (guía, sección 6.1): (1) su AUC medio del bucle
# externo supera al de la mejor combinación de cada uno de los siete modelos, (2) al menos tres pruebas
# distintas rechazan la igualdad de desempeño con p ajustado por comparaciones múltiples menor que 0,05
# (0,10 en el MCS) y (3) el tamaño del efecto no es despreciable. Este capítulo aplica ese protocolo y
# reporta también las pruebas que no rechazan.
#
# ## 6.1 Protocolo
#
# * **Universo:** el AJ-KL y las siete mejores combinaciones del capítulo 3 (los benchmarks del curso), con las mismas particiones
#   externas, la misma semilla y el mismo preprocesamiento. Las predicciones de los benchmarks son las del
#   Entregable 2, emparejadas fila por fila con las del AJ-KL. Los dos comparadores recientes del estado del arte
#   (LightGBM y CatBoost, ajustados con el mismo protocolo) se contrastan aparte en la sección 6.8.
# * **Bloques:** el bucle externo cubre cuatro años (k1-k4) y se divide en 48 bloques mensuales; la prueba
#   2025 tiene 52 semanas. Usar meses y semanas en lugar de solo cinco pliegues da potencia a las pruebas
#   no paramétricas sin repetir la búsqueda de hiperparámetros, que tardó 29 horas-CPU para los 140 modelos.
#   Los meses de un mismo pliegue comparten modelo y año, así que no son independientes: la prueba de
#   Friedman se complementa con una permutación restringida al pliegue y las pruebas sobre 2025 usan errores
#   estándar por bloques de semana o bootstrap estacionario.
# * **Ajuste por comparaciones múltiples:** Holm dentro de cada familia de siete comparaciones (AJ-KL frente
#   a cada benchmark); Nemenyi ya controla el error de familia de sus 28 pares; el MCS controla el error de
#   familia por construcción.

# %%
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.io as pio
from IPython.display import Markdown, display
from scipy import stats

sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
pio.renderers.default = "notebook_connected"
import plots as PL

V = "../data/resultados/validacion/"
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)
tabla = pd.read_parquet(V + "tabla_pruebas.parquet")
W = pd.read_parquet(V + "wilcoxon_mensual.parquet")
Dl = pd.read_parquet(V + "delong_2025.parquet")
DM = pd.read_parquet(V + "diebold_mariano_2025.parquet")
NB = pd.read_parquet(V + "nadeau_bengio.parquet")
res = json.loads(open(V + "resumen.json").read())
mens = pd.read_parquet(V + "metricas_mensuales_externo.parquet")
corto = lambda c: c.split(" | ")[0]

# %% [markdown]
# ## 6.2 Criterio 1: AUC medio del bucle externo

# %%
sel = pd.read_csv("../data/resultados/seleccion_mejores.csv")
ejul = [joblib.load(f"../data/resultados/ajkl/{pl}.joblib")["resumen"] for pl in ["k1", "k2", "k3", "k4"]]
sota = {m: [joblib.load(f"../data/resultados/sota/{m}_{pl}.joblib")["resumen"] for pl in ["k1", "k2", "k3", "k4"]] for m in ["LightGBM", "CatBoost"]}
cols = ["auc", "auc_pr", "brier", "log_loss", "ece", "f1", "precision", "recall", "accuracy"]
comp = pd.DataFrame([{"modelo": "AJ-KL (original)", **{c: np.mean([e[c] for e in ejul]) for c in cols},
                      "auc_de": np.std([e["auc"] for e in ejul], ddof=1)}] +
                    [{"modelo": f"{m} (estado del arte)", **{c: np.mean([e[c] for e in v]) for c in cols},
                      "auc_de": np.std([e["auc"] for e in v], ddof=1)} for m, v in sota.items()] +
                    [{"modelo": r.modelo, **{c: getattr(r, f"{c}_media") for c in cols}, "auc_de": r.auc_de}
                     for r in sel.itertuples()]).set_index("modelo")
display(comp.round(4))
mejor = {c: (comp[c].idxmin() if c in ("brier", "log_loss", "ece") else comp[c].idxmax()) for c in cols}
display(Markdown("Modelo con el mejor valor medio externo en cada métrica: " +
                 "; ".join(f"**{c}**: {m}" for c, m in mejor.items()) + "."))

# %% [markdown]
# ## 6.3 Supuestos de cada prueba

# %%
t = mens.pivot(index="bloque", columns="modelo", values="auc")
ORIG = [c for c in t.columns if c.startswith("AJ-KL")][0]
filas = []
for b in sel["combinacion"]:
    d = (t[ORIG] - t[b]).to_numpy()
    filas.append({"benchmark": corto(b), "media ΔAUC": d.mean(), "asimetría": stats.skew(d),
                  "Shapiro-Wilk p": stats.shapiro(d).pvalue, "autocorrelación rezago 1": pd.Series(d).autocorr(1)})
sup = pd.DataFrame(filas).set_index("benchmark")
display(sup.round(4))
display(Markdown(
    "Wilcoxon supone diferencias simétricas alrededor de la mediana; la asimetría de las diferencias mensuales es "
    f"pequeña (entre {sup['asimetría'].min():.2f} y {sup['asimetría'].max():.2f}). La normalidad no es necesaria para "
    "Wilcoxon, pero sí para la t de Nadeau-Bengio, por eso esta última se usa solo como prueba complementaria. La "
    "autocorrelación de rezago 1 de las diferencias mensuales indica cuánto se parecen meses consecutivos: cuando no "
    "es despreciable, los p de Wilcoxon son algo optimistas y la evidencia debe leerse junto con las pruebas que "
    "respetan la dependencia (permutación por pliegue, DeLong con bloques de semana, Diebold-Mariano con varianza HAC)."))

# %% [markdown]
# ## 6.4 Tabla de pruebas de hipótesis

# %%
mostrar = tabla.copy()
mostrar["conclusión"] = np.where(mostrar["rechaza_h0"], "Rechaza H0", "No rechaza H0")
display(mostrar[["prueba", "comparacion", "h0", "supuestos", "estadistico", "p_crudo", "p_ajustado", "efecto", "conclusión"]]
        .round({"estadistico": 3}))
n_comp = len(tabla)
display(Markdown(f"Se reportan {n_comp} contrastes en total: 1 de Friedman, 7 de Nemenyi, 7 de Wilcoxon, 7 de DeLong, "
                 "2 MCS y 7 de Diebold-Mariano, además de las 7 t de Nadeau-Bengio y las 28 pruebas bayesianas de las "
                 "tablas siguientes."))

# %% [markdown]
# ## 6.5 Diagrama de diferencia crítica

# %%
PL.diagrama_cd(pd.Series({corto(k): v for k, v in res["friedman"]["rangos"].items()}), res["friedman"]["CD"],
               "Friedman-Nemenyi: rango medio del AUC en 48 meses del bucle externo (1 = mejor)").show()

# %% [markdown]
# ## 6.6 Tamaños del efecto e intervalos
#
# Se reportan dos versiones del delta de Cliff porque responden preguntas distintas. El **delta de Cliff
# de dos muestras** {cite}`cliff1993` compara la distribución de los 48 AUC mensuales del AJ-KL con la del
# benchmark como si fueran muestras independientes; la variación entre meses (choques, estacionalidad) es
# mucho mayor que la diferencia entre modelos y lo vuelve pequeño. La **dominancia pareada** es la media de
# signo(ΔAUC) mes a mes: mide qué tan consistente es la ventaja cuando se compara en el mismo mes. La guía
# acepta también la d de Cohen pareada, $d_z$ = media / desviación estándar de las diferencias.

# %%
w = W[W.metrica == "auc"].copy()
w["benchmark"] = w["benchmark"].map(corto)
display(w[["benchmark", "media_dif", "ic_inf", "ic_sup", "meses_gana", "p_holm", "cliff", "mag_cliff",
           "cliff_pareado", "mag_cliff_pareado", "d_cohen", "mag_d", "bayes_p_izq", "bayes_p_rope", "bayes_p_der"]]
        .round(4).set_index("benchmark"))
otros = W[W.metrica != "auc"].assign(benchmark=lambda d: d["benchmark"].map(corto))
display(otros.pivot(index="benchmark", columns="metrica", values="meses_gana").rename(columns=lambda c: f"meses que gana ({c})"))

# %% [markdown]
# La prueba bayesiana de rangos con signo {cite}`benavoli2014signedrank,benavoli2017bayesian` usa una región
# de equivalencia práctica (ROPE) de ±0,01 en AUC, la frontera entre diferencias «triviales» y «pequeñas» de
# la escala del curso. Las columnas `bayes_p_*` son las probabilidades posteriores de que el benchmark sea
# mejor, de que la diferencia caiga en la ROPE y de que el AJ-KL sea mejor.
#
# ## 6.7 Diferencias en la prueba 2025 (intervalos BCa por bloques de semana)

# %%
d = Dl.copy()
d["benchmark"] = d["benchmark"].map(corto)
for m_ in ["auc", "auc_pr", "brier", "log_loss", "ece"]:
    d[m_] = [f"{a:+.4f} [{b:+.4f}; {c:+.4f}]" for a, b, c in zip(d[f"{m_}_dif"], d[f"{m_}_ic_inf"], d[f"{m_}_ic_sup"])]
display(d.set_index("benchmark")[["auc", "auc_pr", "brier", "log_loss", "ece", "p_delong_iid", "p_delong_bloques", "p_holm"]])
display(Markdown("Las diferencias son AJ-KL − benchmark: positivas son mejores en AUC y AUC-PR; negativas son mejores en "
                 "Brier, log-loss y ECE."))

# %%
nb = NB.assign(benchmark=lambda x: x["benchmark"].map(corto)).set_index("benchmark")
display(nb.round(4))
dm = DM.assign(benchmark=lambda x: x["benchmark"].map(corto))
display(dm.pivot(index="benchmark", columns="perdida", values="p_holm").round(4).rename(columns=lambda c: f"p Holm DM ({c})"))

# %% [markdown]
# ## 6.8 Comparación con el estado del arte reciente
#
# LightGBM {cite}`ke2017lightgbm` y CatBoost {cite}`prokhorenkova2018catboost` son los GBDT que la literatura
# reciente sobre datos tabulares sigue señalando como referencia {cite}`shwartzziv2022tabular,borisov2024tabular`.
# No se vieron en clase; se ajustaron con exactamente el mismo protocolo que los benchmarks (mismos pliegues,
# misma muestra $S_k$, TPE con 24 evaluaciones y semilla 42). La tabla resume, para cada uno, el AUC medio del
# bucle externo, la diferencia en 2025 con DeLong por bloques de semana y los meses del bucle externo en que el
# AJ-KL gana.

# %%
import os
if os.path.exists(V + "comparadores_sota.parquet"):
    display(pd.read_parquet(V + "comparadores_sota.parquet").round(5))

# %% [markdown]
# ## 6.9 Comparación con información igual: modelos de producción en 2025
#
# En la validación anidada, el combinador del AJ-KL ve las predicciones fuera de muestra de los bloques
# externos anteriores, más filas etiquetadas que la muestra $S_k$ de 40 mil filas de los benchmarks (aunque
# ninguna posterior a su corte de entrenamiento). La sección 5.8 muestra que reducir ese historial a 40 mil
# filas no cambia el resultado. Una segunda forma de igualar la información es la de producción (Bloque B):
# todos los modelos se reentrenan con las 622.244 filas de entrenamiento, y el AJ-KL combina esos nueve
# artefactos con el combinador del pliegue final, entrenado con predicciones de semanas que los benchmarks
# también usaron para entrenar. Cada artefacto se evalúa una sola vez en las 103.466 filas de 2025.

# %%
if os.path.exists(V + "produccion_2025.parquet"):
    pr = pd.read_parquet(V + "produccion_2025.parquet").assign(benchmark=lambda x: x["benchmark"].map(corto))
    for m_ in ["auc", "auc_pr", "brier", "log_loss", "ece"]:
        pr[m_] = [f"{a:+.4f} [{b:+.4f}; {c:+.4f}]" for a, b, c in zip(pr[f"{m_}_dif"], pr[f"{m_}_ic_inf"], pr[f"{m_}_ic_sup"])]
    display(pr.set_index("benchmark")[["auc_orig", "auc_bench", "auc", "p_delong_holm", "mcnemar_orig_acierta",
                                        "mcnemar_bench_acierta", "p_mcnemar_holm", "brier", "log_loss", "ece"]].round(5))
    art = pd.read_csv("../data/resultados/artefactos_resumen.csv")
    display(art[["modelo", "auc_2025", "auc_pr_2025", "brier_2025", "log_loss_2025", "ece_2025", "f1_2025", "mb", "ms_por_1000"]]
            .sort_values("auc_2025", ascending=False).round(4).set_index("modelo"))

# %% [markdown]
# ## 6.10 Veredicto

# %%
rm = json.loads(open("../data/dashboard/resumen_modelos.json").read())
display(Markdown(rm["veredicto"]))
