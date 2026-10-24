# %% [markdown]
# # 2. Datos: ETL y EDA
#
# Este capítulo resume el ETL de los entregables anteriores y amplía el EDA con lo que necesita el
# modelo original: la distribución de la razón continua de abastecimiento, de la que sale la etiqueta.
# Las figuras usan las mismas funciones que el tablero (`src/plots.py`), de modo que lo que se interpreta
# aquí es lo que se ve en la sección EDA del dashboard.
#
# ## 2.1 ETL
#
# Los microdatos del componente de abastecimiento del SIPSA {cite}`dane_sipsa` registran cada carga que
# ingresa a las centrales mayoristas: fecha, central, municipio de origen, producto y toneladas. El ETL del
# Entregable 1 agregó los 15,5 millones de registros a un panel semanal de corredores (municipio de origen
# → central), descartó los corredores sin historia suficiente para calcular un promedio de 8 semanas y
# construyó variables con información disponible al cierre de la semana *t*: escala y volatilidad del
# corredor, razones de abastecimiento recientes, composición de la carga, calendario, caída simultánea de
# la central y del país y, desde el Entregable 2, variables de vecindad espacial.
#
# El objetivo es
#
# $$
# Y_{i,t+1} = \mathbb{1}\left\{ R_{i,t+1} < 0{,}5 \right\},\qquad
# R_{i,t+1} = \frac{\text{toneladas por día del corredor } i \text{ en } t+1}{\text{promedio de las 8 semanas previas}} ,
# $$ (eq-objetivo)
#
# con $R = 0$ cuando el corredor no envía nada. La partición es cronológica: entrenamiento con las semanas
# objetivo hasta el 27 de octubre de 2024 y prueba con todas las semanas de 2025, separadas por 10 semanas.

# %%
import json
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import Markdown, display

sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
pio.renderers.default = "notebook_connected"
import data as DT
import plots as PL

R = DT.resumen_eda()
m = DT.muestra()
s = DT.semanal()
ETQ = dict(zip(DT.diccionario()["variable"], DT.diccionario()["descripcion"]))
display(pd.DataFrame({"Cifra": ["Registros crudos", "Corredor-semanas", "Corredores", "Municipios de origen", "Centrales",
                                "Semanas", "Filas de entrenamiento", "Filas de prueba (2025)", "Tasa de interrupción"],
                      "Valor": [f"{R['registros_crudos']:,}", f"{R['filas']:,}", f"{R['corredores']:,}", R["municipios"],
                                R["centrales"], R["semanas"], f"{R['filas_entrenamiento']:,}", f"{R['filas_prueba']:,}",
                                f"{100 * R['tasa_interrupcion']:.1f} %"]}).set_index("Cifra"))

# %% [markdown]
# ## 2.2 La variable continua detrás de la etiqueta
#
# La figura siguiente muestra la distribución de $R$ en la muestra estratificada de 80 mil
# corredor-semanas (10 mil por año). Tiene una masa puntual en cero (el corredor no envía nada) y una
# parte continua con moda cerca de 1. La línea vertical marca el umbral de 0,5 que define la interrupción.

# %%
r = m["ratio_t1"].clip(upper=3)
fig = go.Figure(go.Histogram(x=r, xbins=dict(start=0, end=3, size=0.05), marker_color=PL.AZUL,
                             hovertemplate="R en [%{x}]<br>%{y} corredor-semanas<extra></extra>"))
fig.add_vline(x=0.5, line_color=PL.ROJO, line_width=2, annotation_text="τ = 0,5 (interrupción)", annotation_position="top right")
fig.add_vrect(x0=0.2, x1=0.8, fillcolor=PL.NARANJA, opacity=0.12, line_width=0, annotation_text="zona de umbrales cercanos a 0,5",
              annotation_position="top left")
fig.update_layout(title="Distribución de la razón de abastecimiento de t+1 (valores mayores que 3 agrupados en 3)", height=380)
fig.update_xaxes(title="R = abastecimiento en t+1 / promedio de 8 semanas").update_yaxes(title="Corredor-semanas")
fig.show()
cero, bajo, banda = (m["ratio_t1"] == 0).mean(), (m["ratio_t1"] < 0.5).mean(), m["ratio_t1"].between(0.3, 0.7).mean()
display(Markdown(f"El {100 * cero:.1f} % de las corredor-semanas tiene $R = 0$ y el {100 * bajo:.1f} % tiene $R < 0{{,}}5$: "
                 f"el {100 * cero / bajo:.0f} % de las interrupciones son paradas completas. El {100 * banda:.1f} % cae entre 0,3 y 0,7, la franja "
                 "donde la etiqueta binaria es más frágil: un corredor en 0,49 cuenta como interrupción y uno en 0,51 como "
                 "normal. El modelo original aprovecha esa franja entrenando clasificadores en umbrales vecinos de 0,5 "
                 "(capítulo 5)."))

# %% [markdown]
# ## 2.3 Objetivo, desbalance y evolución temporal

# %%
PL.balance_clases(m).show()
PL.serie_tasa(s).show()
anual = m.groupby("anio_t1")["interrupcion_t1"].mean()
display(Markdown(f"La tasa de interrupción es estable entre años ({100 * anual.min():.1f} % a {100 * anual.max():.1f} %), con "
                 f"un máximo en 2021 (paro nacional) y el valor más bajo en 2025. La serie semanal sube en los choques comunes, "
                 "lo que explica la dependencia entre corredores de una misma semana y obliga a validar en orden temporal y a "
                 "remuestrear por bloques de semana en las pruebas estadísticas."))

# %% [markdown]
# ## 2.4 Desigualdad territorial
#
# Las interrupciones no se reparten igual. Los corredores largos y los que salen de municipios PDET
# (Programas de Desarrollo con Enfoque Territorial, priorizados por el conflicto armado) se interrumpen
# con más frecuencia. Esta es la razón por la que el capítulo 5 reporta el desempeño del modelo original
# por subgrupo y por la que el tablero advierte que una alerta no debe usarse para sancionar territorios.

# %%
q = pd.qcut(m["dist_km"], 4, labels=["Q1 (más cortos)", "Q2", "Q3", "Q4 (más largos)"])
t = pd.concat([m.groupby(q, observed=True)["interrupcion_t1"].mean().rename("tasa").to_frame().assign(grupo="Distancia"),
               m.groupby(m["pdet_origen"].map({0: "No PDET", 1: "PDET"}))["interrupcion_t1"].mean().rename("tasa").to_frame().assign(grupo="Origen")])
display((100 * t["tasa"]).round(1).to_frame("Tasa de interrupción (%)").assign(Grupo=t["grupo"]))

# %% [markdown]
# ## 2.5 Variables, faltantes y atípicos

# %%
numericas = [c for c in ["ratio_t", "ratio_t_1", "log_base8", "cv8", "pend8_rel", "activas8", "activas26", "n_envios8",
                         "n_alimentos8", "antig_sem", "dist_km", "frac_caida_mercado_t", "frac_caida_nacional_t",
                         "frac_caida_mpio_t", "frac_caida_vec_t", "tasa_int_vec26", "tasa_int_mpio26", "ratio_t1"]]
PL.correlacion(m, numericas, ETQ).show()
c = m[numericas].corr(method="spearman")["ratio_t1"].drop("ratio_t1").abs().sort_values(ascending=False)
display(Markdown(f"Ninguna variable se asocia fuertemente con la razón de la semana siguiente: la mayor correlación de "
                 f"Spearman en valor absoluto es {c.iloc[0]:.2f} («{ETQ.get(c.index[0])}»). El abastecimiento de t+1 depende "
                 "de choques que no se observan en t, y ese techo de información limita a todos los modelos."))
falt = DT.faltantes()
PL.mapa_faltantes(falt, ETQ).show()
atip = DT.atipicos().sort_values("pct_atipicos", ascending=False)[["etiqueta", "q1", "mediana", "q3", "pct_atipicos"]]
display(atip.head(8).round(3).rename(columns={"etiqueta": "Variable", "pct_atipicos": "% fuera de 1,5 × RIC"}))

# %% [markdown]
# Los faltantes son estructurales y se concentran en una variable: la fracción en caída de los otros
# corredores del mismo municipio (14 % a 16 % de las filas cada año) no está definida cuando el municipio
# tiene un solo corredor activo esa semana. Las variables de vecinos faltan en menos del 1,5 % de las filas
# (vecinos sin corredores activos). Todas se imputan con la mediana de entrenamiento dentro de cada tubería. Los
# atípicos son valores reales: corredores muy grandes o muy volátiles. Los árboles los toleran y los
# modelos lineales los reciben winsorizados al 1 % y al 99 %, como en el Entregable 2.
#
# ## 2.6 Serie de tiempo: descomposición, autocorrelación y estacionariedad

# %%
from statsmodels.tsa.stattools import adfuller, kpss
tt = s.groupby("semana_t1")[["corredores", "interrupciones"]].sum()
serie = (100 * tt["interrupciones"] / tt["corredores"]).asfreq("7D").interpolate()
PL.descomposicion(serie).show()
PL.acf_pacf(serie).show()
adf = adfuller(serie, autolag="AIC")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kp = kpss(serie, regression="c", nlags="auto")
display(pd.DataFrame({"Prueba": ["ADF (H0: raíz unitaria)", "KPSS (H0: estacionaria en nivel)"],
                      "Estadístico": [round(adf[0], 3), round(kp[0], 3)], "p": [round(adf[1], 4), round(kp[1], 4)]}))

# %% [markdown]
# La prueba ADF rechaza la raíz unitaria, mientras que la KPSS rechaza la estacionariedad en nivel: la
# tasa no deriva sin control, pero su nivel cambia por tramos (la tendencia de la descomposición STL baja
# después de 2021). Ese comportamiento es el que motivó, en el capítulo 5, la codificación temporal del
# modelo original: una tasa histórica del corredor calculada con semanas del mismo periodo se desactualiza
# cuando el nivel cambia.
