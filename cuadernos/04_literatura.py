# %% [markdown]
# # 4. Revisión de literatura y novedad
#
# La novedad del modelo original no se declara: se contrasta con una búsqueda documentada (guía, sección
# 5.2). Este capítulo presenta el protocolo, las familias de modelos que encontró la búsqueda, la matriz de
# novedad, la tabla de diferenciación por familia algorítmica y la declaración de novedad.
#
# ## 4.1 Protocolo
#
# * **Problema de aplicación:** predicción de interrupciones semanales de corredores de abastecimiento de
#   alimentos (municipio de origen → central mayorista) con los microdatos del SIPSA-A (DANE, 2018-2025).
#   Tarea: clasificación binaria.
# * **Familia del modelo propuesto:** combinación de clasificadores con pesos locales o jerárquicos
#   (apilamiento jerárquico, selección dinámica de ensambles, combinación de pronósticos de probabilidad,
#   agregación con pesos exponenciales).
# * **Bases:** Web of Science Core Collection y Scopus (ambas por el proxy de la Universidad del Norte) y
#   ScienceDirect, consultadas el 23 de octubre de 2026. ScienceDirect no admite comodines y permite como
#   máximo 8 conectores por campo, así que sus cadenas se acortaron.
# * **Cadenas:** seis por base (una en español) que combinan la familia del modelo, el problema y sinónimos,
#   y una cadena de método puro (M1) para la tabla de diferenciación.
# * **Revisión:** título y resumen de los resultados (hasta 100 por cadena en las de método) y texto completo
#   de los trabajos más cercanos.

# %%
import sys
import warnings

import pandas as pd
from IPython.display import Markdown, display

sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.width", 220)
matriz = pd.read_csv("../data/revision/matriz_novedad_equipo.csv")

# %% [markdown]
# ## 4.2 Familias de modelos mejorados encontradas
#
# | Familia | Qué mejora frente a los benchmarks | Por qué encaja en este problema | Referencias clave |
# |---|---|---|---|
# | A. Apilamiento con pesos heterogéneos o jerárquicos | Los pesos del ensamble cambian según la región de los datos, con agrupamiento parcial entre regiones | El desempeño de cada modelo cambia entre centrales y grupos de alimentos | {cite:t}`yao2022hierarchical`; {cite:t}`kejzlar2023dirichlet`; {cite:t}`huang2025stacking`; {cite:t}`naimi2018superlearner` |
# | B. Combinación de pronósticos de probabilidad | El pool lineal y su versión transformada con beta (BLP) corrigen la falta de calibración del promedio | La salida es una probabilidad de interrupción | {cite:t}`ranjan2010combining`; {cite:t}`gneiting2013combining` |
# | C. Selección dinámica de ensambles | Elige o pondera clasificadores según su competencia local | Ofrece la idea de región de competencia y un comparador directo | {cite:t}`ko2008des`; {cite:t}`woloszynski2011competence`; {cite:t}`cruz2018des`; {cite:t}`abbasi2026des` |
# | D. Agregación con pesos exponenciales | Actualizaciones multiplicativas con cotas de arrepentimiento | Aporta la regla de actualización y su justificación teórica | {cite:t}`cesabianchi2007expert`; {cite:t}`vovk2009brier`; {cite:t}`wintenberger2017boa`; {cite:t}`berrisch2023crps` |
# | E. Árboles y boosting con efectos mixtos | Modelan la heterogeneidad por grupo dentro del aprendiz | El panel corredor-semana es agrupado y longitudinal | {cite:t}`sela2012reem`; {cite:t}`hajjem2014merf`; {cite:t}`fokkema2018glmm`; {cite:t}`salditt2023boosting`; {cite:t}`sigrist2022gpboost` |
# | F. Aprendizaje bajo deriva temporal | Reponderan las observaciones según su antigüedad | Se entrena hasta octubre de 2024 y se prueba en 2025 | {cite:t}`scholz2007drift`; {cite:t}`kanamori2009importance` |
# | Estado del arte tabular (solo como comparador) | Los GBDT siguen siendo la referencia en datos tabulares | Justifica comparar con LightGBM y CatBoost además de XGBoost | {cite:t}`shwartzziv2022tabular`; {cite:t}`borisov2024tabular` |
#
# La guía admite expresamente como contribución original un «ensamble con meta-aprendiz o regla de
# ponderación diseñados y programados por el estudiante (pesos según la región del espacio de entrada, la
# incertidumbre o el error local), con regla de actualización derivada y contrastada contra reglas
# estándar» (cuadro 1). El AJ-KL combina las familias A, B, D y F en esa vía.
#
# ## 4.3 Matriz de novedad

# %%
for base, g in matriz.groupby("base", sort=False):
    display(Markdown(f"**{base}**"))
    display(g[["cadena", "consulta", "resultados", "revisados", "relevantes", "observacion"]].set_index("cadena"))
tot = matriz.groupby("base")[["resultados", "revisados", "relevantes"]].sum()
display(tot)

# %% [markdown]
# Los «relevantes» de las cadenas de método (M1) son trabajos de la misma familia en otras aplicaciones, y el
# de las cadenas D3 y D6 es un trabajo del mismo sistema de abastecimiento con otra tarea (inferencia causal).
# Ninguno aplica una regla de combinación local o jerárquica de clasificadores a la predicción de
# interrupciones de corredores.
#
# **Búsqueda complementaria en Web of Science.** El 23 y 24 de octubre se hizo además una búsqueda por
# aplicación en WoS con 20 cadenas (alerta temprana de seguridad alimentaria, disrupciones de cadenas
# agroalimentarias, llegadas a mercados mayoristas, flujos de carga y SIPSA, en inglés y español). De 464
# resultados se revisaron 363 con doble revisor; ninguno predice interrupciones de corredores de abastecimiento
# con un ensamble jerárquico. La tabla completa está en `data/revision/matriz_novedad_wos.csv` y confirma, por
# el lado de la aplicación, el resultado de la matriz anterior.

# %%
wos = pd.read_csv("../data/revision/matriz_novedad_wos.csv")
display(Markdown(f"Búsqueda complementaria: {len(wos)} cadenas, {int(wos['resultados'].sum())} resultados, "
                 f"{int(wos['revisados'].sum())} revisados, {int(wos['relevantes'].sum())} relevantes."))

# %% [markdown]
# ## 4.4 Trabajos más cercanos
#
# * **Por aplicación.** {cite:t}`nino2026disruption` estima con variables instrumentales el efecto causal de
#   los cierres viales por deslizamientos sobre los precios de los mercados mayoristas colombianos y usa bosques
#   causales para ver en qué mercados el efecto es mayor; no predice interrupciones ni combina modelos. RIOT
#   {cite}`zhang2027riot` combina probabilidades con transporte óptimo para predecir retrasos en cadenas de
#   suministro, con una regla de combinación global, y P-WEV {cite}`singh2026pwev` pondera con PSO modelos GARCH
#   para la volatilidad de precios agrícolas, una tarea de regresión.
# * **Por método.** El apilamiento jerárquico bayesiano {cite}`yao2022hierarchical` es el trabajo más cercano:
#   propone pesos de apilamiento que varían con los datos y se agrupan parcialmente. {cite:t}`yazdi2026shrinkage`
#   compara el encogimiento jerárquico de Bayes empírico con GBDT en la demanda de distribución minorista y
#   encuentra que el peso convexo global se va por completo al GBDT, aunque el encogimiento gana en un estrato,
#   lo que respalda los pesos locales y advierte que la ganancia de combinar puede ser nula.
#
# ## 4.5 Tabla de diferenciación por familia algorítmica
#
# | Método afín | Familia | Qué se conserva | Qué es nuevo en el AJ-KL | Por qué el método existente no resuelve la limitación |
# |---|---|---|---|---|
# | Apilamiento / Super Learner {cite}`naimi2018superlearner` | Combinación lineal | Combinación convexa de probabilidades fuera de muestra | Pesos por nodo de una jerarquía, con encogimiento y olvido | Un solo vector de pesos ignora la heterogeneidad entre centrales, grupos y corredores |
# | BLP {cite}`ranjan2010combining,gneiting2013combining` | Pool de probabilidades | Pool lineal y calibración | Pesos locales con prior jerárquico | Es global; corrige la calibración pero no la heterogeneidad |
# | KNORA / DES {cite}`ko2008des,cruz2018des` | Selección dinámica | Noción de región de competencia | Regiones definidas por la estructura del problema y pesos suaves con objetivo convexo | La competencia por vecinos escala mal a cientos de miles de filas, es inestable donde hay pocos datos y no maneja la deriva |
# | Competencia probabilística {cite}`woloszynski2011competence`; DESbADE {cite}`abbasi2026des` | Selección dinámica | Competencia estimada localmente | Encogimiento hacia el padre en lugar de heurísticas o metaheurísticas sobre la región | Las regiones se construyen sin jerarquía ni agrupamiento parcial |
# | Apilamiento jerárquico bayesiano {cite}`yao2022hierarchical` | Apilamiento jerárquico | Agrupamiento parcial de pesos que varían con los datos | Jerarquía anidada de cuatro niveles con KL hacia el padre, olvido γ y estimador convexo con actualización cerrada en O(n) por iteración | Usa inferencia MCMC con pesos softmax-lineales en covariables, costosa a esta escala y sin olvido explícito |
# | Encogimiento de Bayes empírico frente a GBDT {cite}`yazdi2026shrinkage` | Encogimiento jerárquico | Encoger las estimaciones de grupos pequeños hacia su nivel superior | El encogimiento se aplica a los pesos del ensamble, no a la media del objetivo | Un solo peso global no captura ventajas locales |
# | Mezcla de Dirichlet local {cite}`kejzlar2023dirichlet` | Mezcla bayesiana | Pesos locales | Objetivo frecuentista convexo con jerarquía | Diseñada para modelos físicos con incertidumbre, no para clasificación en un panel |
# | Pesos exponenciales y BOA {cite}`cesabianchi2007expert,wintenberger2017boa,berrisch2023crps` | Agregación en línea | Actualización multiplicativa y olvido | Pesos estructurados por región con prior del padre, aprendidos por lotes dentro de la validación anidada | Pesos globales y secuenciales, sin estructura jerárquica |
# | MERF, RE-EM, GPBoost {cite}`hajjem2014merf,sela2012reem,sigrist2022gpboost` | Efectos mixtos | Modelar la heterogeneidad por grupo | La heterogeneidad se modela en la combinación y aprovecha nueve modelos diversos | Es un solo aprendiz; no combina la diversidad de los modelos |
# | Boosting con deriva {cite}`scholz2007drift` | Aprendizaje con deriva | Ponderación temporal | El olvido actúa en el combinador y se cruza con la jerarquía | No es un combinador local |
#
# ## 4.6 Cambios entre la propuesta y la implementación
#
# La propuesta de la revisión combinaba los 7 modelos del Entregable 2 con una jerarquía de tres niveles
# (país, central y central × grupo). La implementación final (capítulo 5) mantiene el mecanismo y cambia tres
# cosas, por las razones que documenta la sección 5.8: (i) el conjunto de modelos base incluye también
# LightGBM y CatBoost, los dos comparadores recientes ajustados con el mismo protocolo; (ii) la jerarquía añade
# un cuarto nivel, el corredor; y (iii) el combinador se entrena con las predicciones fuera de muestra de los
# bloques externos anteriores, que provienen de modelos entrenados en el mismo régimen que los que se combinan.
# Ninguno de los cambios toca la declaración de novedad: siguen siendo una regla de combinación jerárquica
# con encogimiento KL y olvido, programada por el equipo.
#
# ## 4.7 Declaración de novedad
#
# Al mejor de nuestro conocimiento y según la búsqueda descrita en Web of Science Core Collection, Scopus y
# ScienceDirect (23 de octubre de 2026; cadenas W1-W6, S1-S6, D1-D6 y las cadenas de método M1), complementada
# con 20 cadenas por aplicación en Web of Science (23 y 24 de octubre de 2026), no se encontraron estudios que
# apliquen un apilamiento jerárquico con encogimiento KL y olvido temporal, ni ningún otro esquema de
# combinación local de clasificadores, a la predicción de interrupciones semanales de corredores de
# abastecimiento de alimentos. Los trabajos más cercanos por aplicación son {cite:t}`nino2026disruption`, que
# estima con métodos causales el efecto de los cierres viales sobre los mercados mayoristas colombianos sin
# predecir interrupciones, y RIOT y P-WEV, que usan reglas de combinación globales en tareas distintas. Por
# método, los más cercanos son el apilamiento jerárquico bayesiano {cite}`yao2022hierarchical` y el
# encogimiento jerárquico de Bayes empírico frente a GBDT en logística de distribución
# {cite}`yazdi2026shrinkage`. Nuestra propuesta se diferencia en que estima los pesos del ensamble sobre una
# jerarquía anidada propia del problema, con penalización KL hacia el nodo padre y olvido temporal, mediante un
# estimador convexo con actualización multiplicativa de costo lineal.
