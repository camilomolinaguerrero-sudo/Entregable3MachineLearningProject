# %% [markdown]
# # 5. Modelo original: Ensamble Jerárquico de Umbrales Latentes (EJUL)
#
# ## 5.1 Motivación
#
# La etiqueta del problema no se observa directamente: se construye al dicotomizar una variable
# continua. Para cada corredor-semana se conoce la razón
#
# $$
# R = \frac{\text{abastecimiento en } t+1}{\text{promedio de las 8 semanas previas}}
# $$ (eq-razon)
#
# y la interrupción es el evento $Y = \mathbb{1}\{R < \tau\}$ con $\tau = 0{,}5$. Los siete benchmarks del
# Entregable 2 aprenden solo de $Y$: para ellos, un corredor que cayó a 0,49 y otro que cayó a 0,05
# son igual de «interrumpidos», y uno que quedó en 0,51 es igual de «normal» que otro en 1,2. Dicotomizar
# una variable continua descarta información y reduce la potencia {cite}`altman2006dichotomising`.
#
# El EJUL recupera parte de esa información sin cambiar la tarea. En lugar de un clasificador para el
# umbral $\tau$, entrena varios para umbrales cercanos, $c_k = \tau + h u_k$, cada uno con su propia
# etiqueta $\mathbb{1}\{R < c_k\}$. Cada miembro estima un punto de la función de distribución
# condicional $F(c \mid x) = P(R < c \mid x)$, y la probabilidad pedida es $F(\tau \mid x)$. Los miembros
# vecinos ven etiquetas distintas (un corredor en 0,45 es positivo para $c = 0{,}5$ y negativo para
# $c = 0{,}4$), de modo que sus errores se correlacionan menos que los de varias semillas de un mismo
# modelo; al combinarlos con pesos de polinomio local en la dirección del umbral se reduce la varianza
# con un sesgo controlado. El mecanismo se completa con tres piezas: una proyección isotónica que
# impone que $F$ sea no decreciente en $c$, una codificación Bayes empírica jerárquica de los grupos
# (corredor ⊂ municipio ⊂ departamento) y una recalibración ajustada con predicciones fuera de pliegue.
#
# ## 5.2 Formulación
#
# Sean $x \in \mathbb{R}^p$ las variables, $R \ge 0$ la razón continua (solo disponible en el
# entrenamiento) y $g$ un clasificador probabilístico base (aquí, el XGBoost de referencia con los
# hiperparámetros que su bucle interno eligió en cada pliegue).
#
# **Umbrales y pesos.** Con $K$ impar, $u_k = -1 + 2(k-1)/(K-1)$ y $c_k = \tau + h u_k$. Los pesos
# son los del estimador de polinomio local de grado $d$ evaluado en $u = 0$ {cite}`fan1996local`:
#
# $$
# \omega = e_1^\top (U^\top W U)^{-1} U^\top W,\qquad U_{kj} = u_k^{\,j},\; j = 0,\dots,d,\qquad W = \operatorname{diag}\{\kappa(u_k)\},
# $$ (eq-pesos)
#
# con núcleo $\kappa$ uniforme, triangular o de Epanechnikov. Por construcción, $\sum_k \omega_k = 1$ y
# $\sum_k \omega_k u_k^{\,j} = 0$ para $j = 1, \dots, d$. Con un diseño simétrico y un núcleo simétrico,
# además, todos los momentos impares se anulan.
#
# **Codificación jerárquica.** Para niveles anidados $l = 1, \dots, L$ (del más fino al más grueso),
# con $s_g$ eventos en $n_g$ filas del grupo $g$, la tasa contraída es la media posterior
# Beta-Binomial con prior centrado en la tasa del grupo padre:
#
# $$
# \tilde p_l(g) = \frac{s_g + m_l\, \tilde p_{l+1}(\operatorname{padre}(g))}{n_g + m_l},\qquad \tilde p_{L+1} = \bar y .
# $$ (eq-eb)
#
# La fuerza $m_l$ se estima por momentos: la varianza entre grupos de las tasas observadas, descontada
# la varianza binomial, iguala $\bar y(1-\bar y)/(m_l+1)$; el hiperparámetro `fuerza_eb` la escala. En el
# entrenamiento la codificación de cada fila se calcula con un modelo que no la vio (ajuste cruzado en
# $V$ pliegues), para no filtrar su propia etiqueta {cite}`micci2001target,pargent2022encoding`. Las
# columnas $\operatorname{logit}\tilde p_l$ se añaden a $x$ y forman $z$.
#
# **Miembros, reordenamiento y combinación.** Cada miembro $g_k$ se entrena con la etiqueta
# $\mathbb{1}\{R < c_k\}$ sobre $z$. El vector $\hat F(x) = (g_1(z), \dots, g_K(z))$ se proyecta sobre el
# cono de vectores no decrecientes,
#
# $$
# \Pi\hat F = \arg\min_{v_1 \le \dots \le v_K} \lVert v - \hat F \rVert_2 ,
# $$ (eq-proyeccion)
#
# que se calcula con la fórmula de mínimo-máximo de la regresión isotónica
# {cite}`robertson1988order`, $(\Pi \hat F)_k = \max_{i \le k} \min_{j \ge k} \operatorname{prom}(\hat F_i, \dots, \hat F_j)$.
# La probabilidad combinada es $\tilde p(x) = \omega^\top \Pi \hat F(x)$.
#
# **Recalibración.** Con las predicciones fuera de pliegue $\tilde p_i^{(-v)}$ del entrenamiento se ajusta
# por Newton-Raphson una regresión logística de $y$ sobre $\operatorname{logit}\tilde p$ (Platt,
# {cite}`platt1999`) o sobre $[\ln\tilde p, -\ln(1-\tilde p)]$ (calibración beta, {cite}`kull2017beta`).
# La salida final es $\hat p(x) = \sigma\big(w_0 + w^\top \phi(\tilde p(x))\big)$.
#
# ## 5.3 Propiedades
#
# Las demostraciones usan supuestos explícitos; cuando un supuesto no se cumple en los datos se dice.
#
# **Proposición 1 (sesgo de la combinación).** Supóngase que $F(\cdot \mid x)$ es tres veces
# diferenciable en un entorno de $\tau$ con tercera derivada acotada, que cada miembro es insesgado
# para su punto, $\mathbb{E}[g_k(z)] = F(c_k \mid x)$, y que no actúa el reordenamiento. Entonces
#
# $$
# \mathbb{E}[\tilde p(x)] - F(\tau \mid x) = \frac{h^2}{2} F''(\tau \mid x) \sum_k \omega_k u_k^2 + O(h^4)
# $$ (eq-sesgo)
#
# para el diseño simétrico; con grado $d = 2$ el término en $h^2$ desaparece y el sesgo es $O(h^4)$.
#
# *Demostración.* Por Taylor, $F(\tau + h u_k) = F(\tau) + h u_k F' + \tfrac{1}{2}h^2u_k^2F'' + \tfrac16 h^3u_k^3F''' + O(h^4)$.
# Al multiplicar por $\omega_k$ y sumar, $\sum\omega_k = 1$ deja $F(\tau)$; $\sum \omega_k u_k = 0$ y
# $\sum\omega_k u_k^3 = 0$ (momentos impares de un diseño simétrico con pesos simétricos) eliminan los
# términos de primer y tercer orden. Queda el de segundo orden, que se anula si además
# $\sum \omega_k u_k^2 = 0$, lo que garantizan los pesos de grado 2. $\square$
#
# La proposición explica un hecho que se observó en los datos antes de formalizarla: con grado 0 la
# probabilidad combinada queda encogida hacia el centro (en los corredores de alto riesgo $F$ es
# cóncava en 0,5 y en los de bajo riesgo es convexa), y su ECE sin recalibrar empeora; con grado 2 el
# ECE crudo baja (sección 5.6).
#
# **Proposición 2 (varianza).** Si los errores $\varepsilon_k = g_k - F(c_k\mid x)$ tienen varianza
# $\sigma^2$ y correlación común $\rho$, entonces $\operatorname{Var}(\tilde p) = \sigma^2\left[\rho + (1-\rho)\lVert\omega\rVert_2^2\right]$.
#
# *Demostración.* $\operatorname{Var}(\omega^\top\varepsilon) = \omega^\top\Sigma\omega$ con
# $\Sigma = \sigma^2[(1-\rho)I + \rho\mathbf{1}\mathbf{1}^\top]$, y $\omega^\top\mathbf{1} = 1$. $\square$
#
# Con pesos uniformes $\lVert\omega\rVert_2^2 = 1/K$. El error cuadrático medio queda
#
# $$
# \operatorname{ECM}(h) \approx \Big(\tfrac{h^2}{2}F''\mu_2\Big)^2 + \sigma^2\big[\rho(h) + (1-\rho(h))\lVert\omega\rVert_2^2\big],\qquad \mu_2 = \textstyle\sum_k\omega_k u_k^2 ,
# $$ (eq-ecm)
#
# donde $\rho(h)$ decrece con $h$ porque los miembros más separados ven etiquetas más distintas. El
# semiancho $h$ equilibra ese intercambio y por eso es el hiperparámetro principal que ajusta TPE.
# Si los miembros fueran idénticos ($\rho = 1$) no habría ganancia: el control con varias semillas del
# mismo XGBoost en la ablación mide cuánto de la mejora se debe solo a promediar.
#
# **Proposición 3 (el reordenamiento no empeora el error).** El vector verdadero
# $F(x) = (F(c_1\mid x), \dots, F(c_K\mid x))$ pertenece al cono $\mathcal{C} = \{v_1 \le \dots \le v_K\}$
# porque $c_1 < \dots < c_K$ y $F$ es una función de distribución. Para todo $\hat F$,
# $\lVert \Pi\hat F - F \rVert_2 \le \lVert \hat F - F\rVert_2$ y, en consecuencia,
# $|\omega^\top(\Pi\hat F - F)| \le \lVert\omega\rVert_2\,\lVert\hat F - F\rVert_2$.
#
# *Demostración.* $\mathcal{C}$ es convexo y cerrado; la proyección euclidiana sobre un convexo
# cerrado es no expansiva, $\lVert\Pi a - \Pi b\rVert \le \lVert a - b\rVert$, y $\Pi F = F$. La segunda
# desigualdad es Cauchy-Schwarz. $\square$
#
# El resultado es la versión $L_2$ y puntual del argumento de
# {cite}`chernozhukov2010rearrangement` para curvas de probabilidad.
#
# **Proposición 4 (codificación jerárquica).** Bajo el modelo $s_g \mid p_g \sim \operatorname{Bin}(n_g, p_g)$,
# $p_g \sim \operatorname{Beta}(m p_0, m(1-p_0))$, la expresión {eq}`eq-eb` es la media posterior
# $\mathbb{E}[p_g \mid s_g]$, que minimiza el riesgo cuadrático bayesiano entre todos los estimadores de
# $p_g$, y se escribe como $\lambda_g \tfrac{s_g}{n_g} + (1-\lambda_g)p_0$ con $\lambda_g = n_g/(n_g + m)$.
#
# *Demostración.* Por conjugación, $p_g \mid s_g \sim \operatorname{Beta}(s_g + m p_0, n_g - s_g + m(1-p_0))$,
# cuya media es $(s_g + m p_0)/(n_g + m)$; la media posterior minimiza
# $\mathbb{E}[(\hat p - p_g)^2 \mid s_g]$. $\square$
#
# Un corredor con pocas semanas hereda casi toda la tasa de su municipio, y uno con muchas conserva
# la propia {cite}`efron1975stein,casella1985eb`. Así se aprovecha la dependencia espacial que el
# profesor señaló en los residuos del Entregable 1 sin crear una columna por corredor.
#
# **Proposición 5 (recalibración).** La log-verosimilitud negativa de la regresión logística con un
# término ridge $\lambda\lVert w\rVert^2/2$, $\lambda > 0$, es estrictamente convexa; Newton-Raphson con
# búsqueda lineal hacia atrás converge al único mínimo y lo hace cuadráticamente cerca de él. Con
# pendiente positiva (Platt) o con $a, b \ge 0$ (beta), la recalibración es monótona y no cambia el AUC.
#
# *Demostración.* El hessiano es $\Phi^\top D\Phi + \lambda I$ con $D = \operatorname{diag}(p_i(1-p_i))$,
# definido positivo. La convergencia global con búsqueda lineal y la cuadrática local son resultados
# estándar del método de Newton para funciones estrictamente convexas y dos veces diferenciables con
# hessiano Lipschitz. La monotonía sigue de que $\sigma$ es creciente. $\square$
#
# ## 5.4 Complejidad
#
# Sea $T_g(n, p)$ el costo de entrenar el clasificador base con $n$ filas y $p$ variables y $P_g(n, p)$
# el de predecir. Con $V$ pliegues de recalibración y $L$ niveles de grupo:
#
# * **Entrenamiento:** $O\big(K(V+1)\,T_g(n, p+L)\big)$ por los miembros (cada pliegue usa
#   $(V-1)n/V$ filas), $O(V n L \log G)$ por la codificación cruzada (ordenamiento de los $G$ códigos)
#   y $O(n K^3)$ por el reordenamiento de las predicciones fuera de pliegue. Para XGBoost con
#   histogramas, $T_g = O(T(np + 2^{\delta} p\, b))$ con $T$ árboles de profundidad $\delta$ y $b$
#   intervalos, de modo que el EJUL cuesta del orden de $K(V+1)$ veces el benchmark.
# * **Inferencia:** $O\big(K P_g(n, p) + nL\log G + nK^3\big)$.
# * **Memoria:** $K$ modelos base más las tablas de la codificación, $O(K\,|g| + G)$.
#
# La sección 5.7 contrasta estas expresiones con tiempos medidos al variar $n$ y $p$.

# %% [markdown]
# ## 5.5 Implementación
#
# La clase `EnsambleUmbralesLatentes` (`src/original_model.py`) hereda de `BaseEstimator` y
# `ClassifierMixin`, de modo que funciona con `clone`, `Pipeline` y la validación anidada. La lógica propia
# usa solo numpy y scipy: pesos de polinomio local, proyección isotónica, codificación Bayes empírica
# (cruzada y temporal) y regresión logística por Newton-Raphson. El clasificador base de cada umbral es un
# estimador cualquiera con `predict_proba`; en el problema SIPSA es el XGBoost de referencia con los
# hiperparámetros de su bucle interno, para que la comparación aísle el aporte del mecanismo.

# %%
import inspect
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import Markdown, display

sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
pio.renderers.default = "notebook_connected"
import original_model as OM
import plots as PL

R = "../data/resultados/"
for f in [OM.pesos_polinomio_local, OM.proyeccion_isotonica, OM.CodificadorJerarquico.tasas_ordenadas]:
    print(inspect.getsource(f))

# %% [markdown]
# ### Pruebas unitarias
#
# `tests/test_original_model.py` verifica que el modelo aprende en datos sintéticos con solución conocida
# (AUC > 0,85 y mejor que el clasificador sin la variable continua), que es reproducible con la misma
# semilla, que rechaza entradas e hiperparámetros inválidos, que es compatible con `clone`, y que cada pieza
# cumple su propiedad: la proyección coincide con la regresión isotónica de scikit-learn y no expande
# distancias (proposición 3), los pesos reproducen polinomios del grado pedido, la regresión logística por
# Newton coincide con scikit-learn, la calibración corrige un modelo deliberadamente descalibrado, la
# contracción jerárquica se comporta como la media posterior (proposición 4) y la codificación temporal no
# usa información del mismo periodo ni del futuro.

# %%
import subprocess
r = subprocess.run([sys.executable, "-m", "pytest", "../tests/test_original_model.py", "-q", "--no-header",
                    "-p", "no:cacheprovider"], capture_output=True, text=True)
print(r.stdout[-800:])

# %% [markdown]
# ### Hiperparámetros elegidos en cada pliegue
#
# TPE ajustó, con 24 evaluaciones y el AUC interno como criterio (el mismo de los benchmarks), el semiancho
# $h$, el número de umbrales $K$, el núcleo, el grado, la fuerza del prior y el uso de la codificación
# jerárquica; el calibrador se eligió después por log-loss interna.

# %%
res = {pl: joblib.load(f"{R}original/{pl}.joblib") for pl in ["k1", "k2", "k3", "k4", "final"]}
display(pd.DataFrame({pl: r["resumen"]["params"] for pl, r in res.items()}).T)
hist = pd.concat([r["historial"].assign(pliegue=pl) for pl, r in res.items()])
PL.convergencia_tpe(hist).show()

# %% [markdown]
# ## 5.6 Ablación y reglas de combinación
#
# Cada fila retira un componente del modelo elegido en cada pliegue (o cambia su regla de combinación) y
# reporta la diferencia media frente al modelo completo en los cinco pliegues externos. Las reglas estándar
# usan los mismos miembros y la misma recalibración: promedio simple, pesos proporcionales al inverso del
# Brier fuera de pliegue y stacking lineal (regresión logística sobre los logit de los miembros). Los dos
# controles miden cuánto de la mejora se obtiene sin el mecanismo: varios XGBoost con distinta semilla
# (reducción de varianza pura) y una regresión de $\log(1+R)$ con residuos empíricos (usar la variable
# continua sin umbrales).

# %%
abl = pd.concat([joblib.load(f"{R}ablacion/{pl}.joblib")["tabla"] for pl in ["k1", "k2", "k3", "k4", "final"]])
met = ["auc", "auc_pr", "brier", "log_loss", "ece", "f1"]
comp = abl[abl.variante == "completo"].set_index("pliegue")[met]
dif = abl.set_index(["variante", "pliegue"])[met] - comp
tabla_abl = dif.groupby("variante").mean()
tabla_abl["pliegues con menor AUC que el completo"] = (dif["auc"] < 0).groupby("variante").sum()
display(tabla_abl.drop("completo").sort_values("auc").round(5))

# %% [markdown]
# ## 5.7 Complejidad empírica

# %%
cx = pd.read_parquet(R + "complejidad_ejul.parquet")
for exp, var in [("n", "n"), ("p", "p")]:
    d = cx[cx.experimento == exp]
    fig = go.Figure()
    for modelo, col in [("XGBoost", PL.AZUL), ("EJUL", PL.NARANJA)]:
        g = d[d.modelo == modelo]
        pend = np.polyfit(np.log(g[var]), np.log(g["t_fit"]), 1)[0]
        fig.add_scatter(x=g[var], y=g["t_fit"], mode="lines+markers", name=f"{modelo} (pendiente log-log {pend:.2f})",
                        line=dict(color=col))
    fig.update_layout(title=f"Tiempo de entrenamiento al variar {'n (filas)' if exp == 'n' else 'p (variables)'}", height=360)
    fig.update_xaxes(title="Filas de entrenamiento" if exp == "n" else "Variables", type="log")
    fig.update_yaxes(title="Segundos (mediana de 3)", type="log")
    fig.show()
d = cx[cx.experimento == "n"].pivot(index="n", columns="modelo", values=["t_fit", "t_pred_1000"])
d["razón entrenamiento"] = d[("t_fit", "EJUL")] / d[("t_fit", "XGBoost")]
d["razón predicción"] = d[("t_pred_1000", "EJUL")] / d[("t_pred_1000", "XGBoost")]
display(d.round(3))

# %% [markdown]
# ## 5.8 Iteraciones del diseño
#
# El diseño final es la segunda versión del modelo. Las iteraciones se reportan completas porque varias se
# decidieron después de ver resultados, y el lector debe poder juzgar cuánto pesa esa información.
#
# | Versión | Cambio | Evidencia que lo motivó | Resultado |
# |---|---|---|---|
# | Exploración | Promedio de umbrales vecinos y codificación del corredor sobre XGBoost | Pruebas rápidas en k1-k4 (`scripts/exploracion/`) | +0,0016 de AUC en los cuatro bloques, pero peor ECE |
# | v1a | Clase completa con recalibración; TPE con log-loss interna como criterio | — | ΔAUC externo medio +0,0010; en 2025, +0,00002 |
# | v1b | Mismo modelo, TPE con AUC interno (el criterio de los benchmarks) | Coherencia con el protocolo del Entregable 2 | ΔAUC externo +0,0010; en 2025, +0,0001; ECE peor en los cinco pliegues |
# | Diagnóstico | Reentrenar con todos los datos | El artefacto v1 con 622 mil filas quedó por debajo de XGBoost en 2025 | Con todos los datos de k3 y k4, v1 empata o pierde y empeora el ECE (tabla siguiente) |
# | v2 (final) | Codificación jerárquica temporal: cada fila de entrenamiento se codifica solo con semanas anteriores | La codificación cruzada aleatoria usa semanas vecinas del mismo periodo y es optimista cuando el nivel cambia (sección 2.6) | Con todos los datos de k3 y k4 supera a XGBoost; resultados del protocolo anidado en el capítulo 6 |
#
# Los resultados de 2025 de v1a y v1b se conocían cuando se diseñó v2. La decisión de pasar a la
# codificación temporal se tomó con los bloques externos k3 y k4 entrenados con todos sus datos, no con
# 2025, y tiene una justificación independiente de los datos: en producción el modelo solo conoce el pasado.

# %%
esc = pd.concat([pd.read_csv(f"{R}iteraciones/escala_{pl}.csv", index_col=0).assign(pliegue=pl) for pl in ["k3", "k4"]])
esc.index.name = "modelo"
display(esc.reset_index().pivot(index="modelo", columns="pliegue", values=["auc", "ece", "ll"]).round(5))

# %% [markdown]
# ## 5.9 Interpretabilidad: SHAP del EJUL frente a XGBoost
#
# Los valores de Shapley se estiman por permutaciones sobre las 34 variables crudas (incluidas las
# categóricas) para 300 corredor-semanas de 2025, con 40 filas de fondo {cite}`strumbelj2014shapley,lundberg2017shap`.
# El mismo estimador explica las predicciones individuales en el tablero.

# %%
sg = pd.read_parquet("../data/dashboard/shap_global.parquet")
piv = sg.pivot(index="variable", columns="modelo", values="importancia")
top = piv.sort_values("ejul_original", ascending=False).head(15)[["ejul_original", "xgboost"]]
fig = go.Figure()
fig.add_bar(y=top.index[::-1], x=top["ejul_original"][::-1], orientation="h", name="EJUL", marker_color=PL.NARANJA)
fig.add_bar(y=top.index[::-1], x=top["xgboost"][::-1], orientation="h", name="XGBoost", marker_color=PL.AZUL)
fig.update_layout(title="Importancia global: media de |SHAP| (puntos de probabilidad), 15 variables principales del EJUL",
                  barmode="group", height=560)
fig.show()
from scipy.stats import spearmanr
rho = spearmanr(piv["ejul_original"], piv["xgboost"]).correlation
display(Markdown(f"La correlación de Spearman entre las importancias de los dos modelos es {rho:.2f}."))

# %% [markdown]
# ## 5.10 Robustez

# %%
import glob
rob = pd.concat([pd.read_parquet(f) for f in glob.glob(R + "robustez/*.parquet")])
rob_w = rob.pivot_table(index=["experimento", "condicion", "valor", "pliegue"], columns="modelo", values=["auc", "ece", "brier"])
dd = pd.DataFrame({"ΔAUC": rob_w[("auc", "ejul")] - rob_w[("auc", "xgb")],
                   "ΔBrier": rob_w[("brier", "ejul")] - rob_w[("brier", "xgb")],
                   "ΔECE": rob_w[("ece", "ejul")] - rob_w[("ece", "xgb")]})
resumen_rob = dd.groupby(["experimento", "condicion", "valor"]).agg(["mean", "min", "max"])
resumen_rob["pliegues que gana (AUC)"] = (dd["ΔAUC"] > 0).groupby(["experimento", "condicion", "valor"]).sum()
display(resumen_rob.round(5))

# %% [markdown]
# ## 5.11 Dónde el EJUL no supera a XGBoost
#
# Desempeño en la prueba 2025 por subgrupos, con las predicciones del pliegue final de la validación anidada.

# %%
import data as DT
import datos as D
panel = D.cargar_panel().set_index("fila")
pru = pd.read_parquet(R + "pred_prueba_clf.parquet")
p = res["final"]["pred"]
pru["EJUL"] = pd.Series(p["s"], index=p["fila"]).reindex(pru["fila"]).to_numpy()
pru["XGBoost"] = pru["XGBoost | Sin balanceo | Random Search"]
info = panel.loc[pru["fila"], ["dist_km", "pdet_origen", "mercado", "depto_origen", "log_base8"]].reset_index(drop=True)
pru = pd.concat([pru.reset_index(drop=True), info], axis=1)
pru["distancia"] = pd.qcut(pru["dist_km"], 4, labels=["Q1 corta", "Q2", "Q3", "Q4 larga"])
pru["escala"] = pd.qcut(pru["log_base8"], 4, labels=["Q1 pequeño", "Q2", "Q3", "Q4 grande"])
pru["PDET"] = pru["pdet_origen"].map({0: "No PDET", 1: "PDET"})
from sklearn.metrics import roc_auc_score
filas = []
for grupo in ["distancia", "escala", "PDET"]:
    for nivel, g in pru.groupby(grupo, observed=True):
        filas.append({"grupo": grupo, "nivel": nivel, "n": len(g), "tasa": g["interrupcion_t1"].mean(),
                      "AUC EJUL": roc_auc_score(g["interrupcion_t1"], g["EJUL"]),
                      "AUC XGBoost": roc_auc_score(g["interrupcion_t1"], g["XGBoost"])})
sub = pd.DataFrame(filas)
sub["ΔAUC"] = sub["AUC EJUL"] - sub["AUC XGBoost"]
display(sub.round(4))
cent = pru.groupby("mercado").apply(lambda g: roc_auc_score(g["interrupcion_t1"], g["EJUL"]) - roc_auc_score(g["interrupcion_t1"], g["XGBoost"])
                                    if g["interrupcion_t1"].nunique() > 1 else np.nan).dropna()
display(Markdown(f"Por central mayorista, el EJUL tiene mayor AUC que XGBoost en {(cent > 0).sum()} de {len(cent)} centrales; "
                 f"la mayor desventaja es {cent.min():+.4f} ({cent.idxmin()}) y la mayor ventaja {cent.max():+.4f} ({cent.idxmax()})."))
