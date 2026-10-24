# %% [markdown]
# # 5. Modelo original: apilamiento jerárquico con encogimiento KL y olvido temporal (AJ-KL)
#
# ## 5.1 Motivación
#
# Los modelos del Entregable 2 quedan a menos de 0,02 de AUC entre sí y XGBoost es el mejor en promedio,
# pero no en todas partes. Su ventaja cambia entre centrales mayoristas, grupos de alimentos y corredores, y
# cambia con el tiempo: un modelo puede ser el mejor para las frutas que llegan a una central de la costa y no
# para los tubérculos que llegan a Bogotá. Un único vector de pesos (el apilamiento clásico) ignora esa
# heterogeneidad; estimar un vector de pesos independiente por corredor la sobreajusta, porque muchos
# corredores tienen pocas semanas.
#
# El AJ-KL toma una posición intermedia con una estructura que ya tiene el problema. Combina las
# probabilidades de los modelos base con pesos que dependen del nodo de una jerarquía
# (país → central → central × grupo de alimentos → corredor). Los pesos de cada nodo se encogen hacia los de
# su nodo padre con una penalización de Kullback-Leibler, de modo que un corredor con pocos datos hereda los
# pesos de su central y uno con muchos datos se aparta de ellos solo si la evidencia lo justifica, y las semanas
# antiguas pesan menos que las recientes. La propuesta sale de la revisión de literatura del capítulo 4, que la
# ubica en la familia de los ensambles con regla de ponderación propia que la guía admite como contribución
# original (cuadro 1).
#
# ## 5.2 Formulación
#
# Sean $i = 1, \dots, n$ las corredor-semanas, $t_i$ su semana, $y_i \in \{0, 1\}$ la interrupción y
# $p_i = (p_{i1}, \dots, p_{iM}) \in [\varepsilon, 1 - \varepsilon]^M$ las probabilidades fuera de muestra de
# los $M$ modelos base. La jerarquía $\mathcal{T}$ tiene raíz $\rho$; cada nodo $r \neq \rho$ tiene un padre
# $\pi(r)$ y un vector de pesos $w_r$ en el símplex
# $\Delta^M = \{w \in \mathbb{R}^M : w_m \ge 0, \sum_m w_m = 1\}$. La predicción para una fila de la hoja $r$
# es un pool lineal
#
# $$
# q_i = w_r^\top p_i .
# $$ (eq-pool)
#
# Los pesos de cada nodo se estiman, de la raíz hacia las hojas, como
#
# $$
# w_r = \arg\min_{w \in \Delta^M} J_r(w),\qquad
# J_r(w) = \sum_{i \in r} \gamma^{\,t_{\max} - t_i}\, \ell(y_i, w^\top p_i) + \lambda\, \mathrm{KL}(w \,\|\, w_{\pi(r)}),
# $$ (eq-objetivo-ajkl)
#
# con la pérdida logarítmica $\ell(y, q) = -y \log q - (1 - y)\log(1 - q)$, el factor de olvido
# $\gamma \in (0, 1]$, la fuerza del encogimiento $\lambda \ge 0$ y, en la raíz, $w_{\pi(\rho)} = u$ la
# distribución uniforme. Al dividir por $S_r = \sum_{i \in r}\gamma^{t_{\max} - t_i}$ (el tamaño de muestra
# efectivo del nodo) el problema queda $f_r(w) + \mu_r \mathrm{KL}(w \| w_{\pi(r)})$ con $f_r$ la pérdida media
# ponderada y $\mu_r = \lambda / S_r$: el encogimiento efectivo crece cuando el nodo tiene pocos datos.
#
# **Regla de actualización.** Cada nodo se resuelve con descenso espejo compuesto con el regularizador
# entrópico {cite}`beck2003mirror,duchi2010composite`. En la iteración $k$, con paso $\eta_k$,
#
# $$
# w^{(k+1)} = \arg\min_{w \in \Delta^M}\; \eta_k \langle \nabla f_r(w^{(k)}), w\rangle + \eta_k \mu_r \mathrm{KL}(w \| w_{\pi(r)}) + \mathrm{KL}(w \| w^{(k)}),
# $$ (eq-comid)
#
# que tiene solución cerrada (proposición 2):
#
# $$
# \log w^{(k+1)}_m = \frac{\log w^{(k)}_m - \eta_k \nabla_m f_r(w^{(k)}) + \eta_k \mu_r \log w_{\pi(r),m}}{1 + \eta_k \mu_r} - c_k,
# \qquad
# \nabla_m f_r(w) = \frac{1}{S_r}\sum_{i \in r} \gamma^{t_{\max} - t_i}\, \frac{q_i - y_i}{q_i(1 - q_i)}\, p_{im},
# $$ (eq-actualizacion)
#
# donde $c_k$ normaliza al símplex. El algoritmo empieza en los pesos del padre. El paso se ajusta por
# retroceso: se acepta si $F_r = f_r + \mu_r\mathrm{KL}(\cdot\|w_{\pi(r)})$ baja (si no, $\eta$ se divide
# entre 2) y después de cada paso aceptado crece un 50 %. La actualización multiplicativa tiene un punto
# ciego: si un peso está en casi cero (por ejemplo, porque el padre casi no usa ese modelo), solo puede crecer
# en proporción a sí mismo y el algoritmo se estanca lejos del óptimo. Por eso, cuando el paso multiplicativo
# mejora $F_r$ en menos de $10^{-12}$, se prueba un paso de Frank-Wolfe {cite}`jaggi2013revisiting`:
# $w \leftarrow (1 - t)\,w + t\, e_j$, con $j = \arg\min_m \nabla_m F_r(w)$ y $t \in [0, 1]$ elegido por
# búsqueda lineal de sección dorada, que sí puede devolverle peso a ese modelo. La parada no usa el tamaño del cambio sino la brecha
# de Frank-Wolfe {cite}`jaggi2013revisiting`,
#
# $$
# g(w) = \langle \nabla F_r(w), w\rangle - \min_m \nabla_m F_r(w),
# $$ (eq-brecha)
#
# que acota el error de optimización (proposición 3): el algoritmo se detiene cuando $g(w) < 10^{-8}$ o
# cuando ningún paso reduce $F_r$, de modo que los pesos devueltos son el minimizador de $J_r$ salvo la
# tolerancia numérica.
#
# *Versión 2.0 y corrección.* La primera implementación usaba $\eta_k = \eta_0/\sqrt{k}$ con $K \le 300$
# iteraciones y paraba por $\|w^{(k+1)} - w^{(k)}\|_1$. La revisión adversarial del código (capítulo 8)
# mostró que con nueve modelos base muy correlacionados el objetivo es casi plano y esos pasos dejaban los
# pesos de la raíz lejos del óptimo ($\|w - w^\star\|_1 = 0{,}53$ en k1): $\eta_0$ y $K$ funcionaban como
# una regularización implícita hacia el padre que la teoría no describía. La versión 2.1, que es la que se
# reporta, resuelve cada nodo hasta el certificado y llega al mismo óptimo que L-BFGS en 106 iteraciones
# (0,05 s) en ese caso; $\eta_0$ y $K$ dejaron de ser hiperparámetros. El paso de Frank-Wolfe se añadió
# después de revisar las brechas por nodo (sección 5.7): sin él, algunos nodos de central cuyo padre daba peso
# casi nulo a un modelo útil se detenían con brechas de hasta $3 \times 10^{-3}$. Una fila
# cuyo nodo no tiene pesos (no se vio en el entrenamiento) usa los de su ancestro más profundo que sí los
# tenga.
#
# **Calibración.** Un pool lineal de pronósticos calibrados no está calibrado salvo que todos coincidan
# {cite}`ranjan2010combining`: queda subdisperso, con probabilidades encogidas hacia el centro. Por eso,
# después del pool se ajusta una calibración beta $\sigma(a \log q - b\log(1 - q) + c)$ {cite}`kull2017beta`
# por Newton-Raphson ponderado con los mismos pesos de olvido. Es una transformación monótona: no cambia el
# AUC y se activa solo si mejora la log-loss en la validación interna.
#
# ## 5.3 Propiedades
#
# **Proposición 1 (convexidad estricta y unicidad).** Para $\lambda > 0$ y $p_i \in [\varepsilon, 1-\varepsilon]^M$,
# $J_r$ es estrictamente convexa en $\Delta^M$ y tiene un único minimizador; si además el padre tiene
# soporte completo ($w_{\pi(r),m} > 0$ para todo $m$), el minimizador está en el interior del símplex.
#
# *Demostración.* $q = w^\top p$ es afín en $w$ y $-\log q$ y $-\log(1 - q)$ son convexas en $q \in (0,1)$, así
# que cada término $\ell(y_i, w^\top p_i)$ es convexo en $w$ y la suma ponderada con pesos positivos también.
# $\mathrm{KL}(w \| v) = \sum_m w_m \log(w_m / v_m)$ es estrictamente convexa en $w$ (su hessiano es
# $\mathrm{diag}(1/w_m) \succ 0$). La suma de una función convexa y una estrictamente convexa es
# estrictamente convexa; $\Delta^M$ es compacto y convexo y $J_r$ es continua, así que existe un minimizador y
# es único. Como $\partial \mathrm{KL}/\partial w_m = \log(w_m/v_m) + 1 \to -\infty$ cuando $w_m \to 0$ y el
# gradiente de la pérdida está acotado (proposición 3), el óptimo no puede tener componentes nulas. $\square$
#
# Con $\lambda = 0$ la convexidad se conserva pero la unicidad no está garantizada: $f_r$ es estrictamente
# convexa en $q = P_r w$, no en $w$, y si dos modelos base son casi colineales en el nodo el conjunto de
# minimizadores es un politopo. Por eso la ablación «λ = 0» se interpreta como un caso límite (proposición 4).
#
# **Proposición 2 (la actualización es exacta y sus puntos fijos son las condiciones KKT).** La expresión
# {eq}`eq-actualizacion` es la solución única del subproblema {eq}`eq-comid`, y $w^\star$ es punto fijo de la
# iteración si y solo si cumple las condiciones de Karush-Kuhn-Tucker del problema del nodo.
#
# *Demostración.* El lagrangiano del subproblema, con multiplicador $\nu$ para $\sum_m w_m = 1$, es
# $\eta \langle g, w\rangle + \eta\mu \sum_m w_m \log(w_m/v_m) + \sum_m w_m \log(w_m / w^{(k)}_m) + \nu(\sum_m w_m - 1)$,
# con $g = \nabla f(w^{(k)})$ y $v = w_{\pi(r)}$. La entropía mantiene $w_m > 0$, así que las restricciones de
# no negatividad no se activan. Igualando a cero la derivada respecto de $w_m$:
# $\eta g_m + \eta\mu(\log w_m - \log v_m + 1) + \log w_m - \log w^{(k)}_m + 1 + \nu = 0$, de donde
# $(1 + \eta\mu)\log w_m = \log w^{(k)}_m - \eta g_m + \eta\mu \log v_m - \text{cte}$, que es
# {eq}`eq-actualizacion` con $c_k$ fijado por la normalización. El subproblema es estrictamente convexo, así
# que la solución es única. Si $w^{(k+1)} = w^{(k)} = w^\star$, la ecuación queda
# $\nabla_m f(w^\star) + \mu(\log w^\star_m - \log v_m + 1) = $ cte para todo $m$, que es exactamente la
# condición de estacionariedad de $f + \mu\mathrm{KL}(\cdot\|v)$ sobre el símplex con multiplicador común; por
# convexidad (proposición 1) ese punto es el óptimo. $\square$
#
# **Proposición 3 (certificado de optimalidad y convergencia).** (a) Para todo $w \in \Delta^M$,
# $F_r(w) - F_r^\star \le g(w)$, con $g$ la brecha {eq}`eq-brecha`. (b) Con $p_i \in [\varepsilon, 1-\varepsilon]^M$,
# $f_r$ es $L$-suave respecto de $\|\cdot\|_1$ con $L \le 1/\varepsilon^2$, y el descenso espejo compuesto con
# paso constante $\eta \le 1/L$ cumple
#
# $$
# F_r(w^{(K)}) - F_r^\star \le \frac{\mathrm{KL}(w_r^\star \,\|\, w_{\pi(r)})}{\eta K}.
# $$ (eq-tasa)
#
# *Demostración.* (a) Por convexidad, $F_r(w) - F_r(w^\star) \le \langle \nabla F_r(w), w - w^\star\rangle$, y como
# $w^\star \in \Delta^M$, $\langle \nabla F_r(w), w^\star\rangle \ge \min_m \nabla_m F_r(w)$; restando queda la
# brecha. (b) El hessiano de $f_r$ es $\frac{1}{S_r}\sum_i a_i \kappa_i\, p_i p_i^\top$ con
# $\kappa_i = y_i/q_i^2 + (1-y_i)/(1-q_i)^2 \le 1/\varepsilon^2$, porque $q_i$ es una combinación convexa de
# valores en $[\varepsilon, 1-\varepsilon]$; para $d \in \mathbb{R}^M$, $(p_i^\top d)^2 \le \|d\|_1^2$ porque
# $p_{im} \le 1$, así que $d^\top \nabla^2 f_r\, d \le \|d\|_1^2/\varepsilon^2$. La entropía negativa es
# 1-fuertemente convexa respecto de $\|\cdot\|_1$ en el símplex (desigualdad de Pinsker), de modo que $f_r$ es
# $L$-suave relativa a ella. El término $\mu_r\mathrm{KL}(\cdot\|w_{\pi(r)})$ se trata de forma exacta en el
# paso (proposición 2), y el resultado de convergencia del método de gradiente de Bregman para funciones
# relativamente suaves {cite}`lu2018relatively` (teorema 3.1, que se extiende al término compuesto como en
# {cite:t}`duchi2010composite`) da {eq}`eq-tasa`, con la divergencia inicial medida desde el punto de partida,
# que es el padre. $\square$
#
# La cota de (b) es solo una garantía en el peor caso: con $\varepsilon = 10^{-4}$, $1/L$ es diminuto. Por eso
# la implementación adapta el paso por retroceso y usa (a) como criterio de parada: no importa cuántas
# iteraciones tome, al detenerse con $g(w) < 10^{-8}$ el error en el objetivo es menor que $10^{-8}$. Con $\mu_r > 0$
# el objetivo es además $\mu_r$-fuertemente convexo relativo a la entropía, de modo que
# $\mathrm{KL}(w^\star\|w) \le g(w)/\mu_r$ también acota la distancia a los pesos óptimos. La sección 5.7 reporta
# las brechas alcanzadas en el modelo final.
#
# **Proposición 4 (casos límite).** (a) Con $\lambda \to 0^+$ cada nodo tiende a un apilamiento local sin
# encogimiento: entre los minimizadores de $f_r$, el más cercano a los pesos del padre en KL.
# (b) Con $\lambda \to \infty$, todos los pesos convergen a la uniforme: el modelo se vuelve el promedio simple.
# (c) Con profundidad 1 (solo la raíz) el modelo es un apilamiento global en el símplex encogido hacia la
# uniforme con $\mu_\rho = \lambda / S_\rho$, que es despreciable cuando $S_\rho \gg \lambda$. (d) Con
# $\gamma = 1$ desaparece el olvido.
#
# *Demostración.* (d) es inmediata de {eq}`eq-objetivo-ajkl`. (a) $f_r$ es convexa y continua en el compacto
# $\Delta^M$ y $\mathrm{KL}(\cdot\|w_{\pi(r)})$ es estrictamente convexa y continua; el resultado clásico de
# regularización de Tikhonov da que los minimizadores de $f_r + \mu\mathrm{KL}$ convergen, cuando $\mu \to 0^+$,
# al elemento de $\arg\min f_r$ que minimiza $\mathrm{KL}(\cdot\|w_{\pi(r)})$. (b) Dividiendo $J_r$ por $\lambda$, el
# problema converge a $\min_w \mathrm{KL}(w \| w_{\pi(r)})$, cuyo único mínimo es $w = w_{\pi(r)}$; por
# inducción desde la raíz (cuyo padre es $u$), todos los nodos tienden a $u$. (c) El término de encogimiento de
# la raíz pesa $\mu_\rho = \lambda / S_\rho \to 0$ cuando $S_\rho$ crece, así que la raíz minimiza la
# log-loss ponderada del pool en el símplex, que es la definición del apilamiento lineal global. $\square$
#
# Las reglas estándar contra las que se contrasta el modelo (promedio simple, apilamiento global y pesos por
# error global) son, por tanto, casos particulares o vecinos del AJ-KL, lo que hace natural la ablación.
#
# **Proposición 5 (interpretación bayesiana del encogimiento).** El problema del nodo es el estimador de máxima
# a posteriori bajo una verosimilitud de Bernoulli con exponente $\gamma^{t_{\max}-t_i}$ (verosimilitud
# ponderada) y el prior entrópico $\pi(w) \propto \exp\{-\lambda \mathrm{KL}(w \| w_{\pi(r)})\}$ centrado en los
# pesos del padre.
#
# *Demostración.* El logaritmo del posterior es
# $-\sum_i \gamma^{t_{\max}-t_i}\ell(y_i, w^\top p_i) - \lambda \mathrm{KL}(w\|w_{\pi(r)}) + $ cte $= -J_r(w) +$ cte. $\square$
#
# Como en el Bayes empírico jerárquico, el peso del prior frente a los datos es $\lambda$ frente a $S_r$: con
# $S_r \gg \lambda$ dominan los datos del nodo; con $S_r \ll \lambda$ el nodo copia a su padre. El olvido reduce
# $S_r$ (el tamaño de muestra efectivo con semivida $\log 2 / \log(1/\gamma)$ semanas), así que los nodos con
# datos antiguos se encogen más: el modelo confía menos en evidencia vieja.
#
# **Proposición 6 (capacidad y generalización).** Fijados los modelos base, la clase de predictores del nodo
# $\{x \mapsto w^\top p(x) : w \in \Delta^M\}$ es la envolvente convexa de $M$ funciones con valores en
# $[0, 1]$, y su complejidad de Rademacher empírica con $n_r$ filas es a lo sumo $\sqrt{2\log M / n_r}$.
#
# *Demostración.* El supremo de una función lineal sobre el símplex se alcanza en un vértice, así que la
# complejidad de Rademacher de la envolvente convexa es igual a la del conjunto finito de los $M$ modelos base;
# el lema de Massart para clases finitas con valores en $[0,1]$ da $\sqrt{2 \log M / n_r}$. $\square$
#
# La cota crece con el logaritmo del número de modelos y no con el número de variables: el combinador tiene muy
# poca capacidad por nodo, y la jerarquía con encogimiento evita que los nodos pequeños (con $n_r$ chico y
# cota grande) se aparten mucho de su padre. Esa es la justificación de usar nodos tan finos como el corredor.
#
# ## 5.4 Complejidad
#
# Sea $L$ el número de niveles, $K$ el número de iteraciones por nodo (incluidos los retrocesos) y $n$ el
# número de filas del historial. En cada nivel, cada fila pertenece a un solo nodo, así que una iteración sobre
# todos los nodos del nivel cuesta $O(nM)$ (evaluar el pool, el objetivo y el gradiente); agrupar las filas por
# nodo cuesta $O(n\log n)$ (un ordenamiento por nivel). El entrenamiento cuesta $O(L\,(n \log n + K\,n\,M))$
# más el costo de los modelos base, que ya están entrenados. La inferencia cuesta $O(L \log N + M)$ por fila,
# con $N$ el número de nodos del nivel (búsqueda binaria del código en la tabla de cada nivel y un producto
# interno de longitud $M$); la memoria es $O(|\mathcal{T}|\, M)$. En el modelo final $K$ fue a lo sumo del
# orden de cien iteraciones por nodo (sección 5.7).
# La sección 5.7 contrasta estas expresiones con tiempos medidos.
#
# ## 5.5 Pseudocódigo
#
# ```text
# Entrada: probabilidades fuera de muestra P (n × M) de los modelos base, etiquetas y, semanas t,
#          códigos de la jerarquía por nivel (central, central × grupo, corredor),
#          hiperparámetros λ, γ, profundidad; tolerancia (10⁻⁸), ε; calibrar ∈ {sí, no}
# Salida:  pesos w_r por nodo; calibrador beta (opcional)
#
# 1. P ← recortar(P, ε, 1 − ε);  a_i ← γ^(t_max − t_i)
# 2. w_raíz ← DescensoEspejo(P, y, a, padre = uniforme, μ = λ / Σ a_i)
# 3. para cada nivel l = 1 .. profundidad − 1:
#        para cada nodo r del nivel l:
#            w_r ← DescensoEspejo(P_r, y_r, a_r, padre = w_{π(r)}, μ = λ / Σ_{i∈r} a_i)
# 4. si calibrar: (a, b, c) ← Newton ponderado de y sobre [log q, −log(1 − q)] con pesos a,
#    donde q_i = w_{nodo(i)}ᵀ p_i
# 5. devolver {w_r}, (a, b, c)
#
# DescensoEspejo(P, y, a, padre, μ):
#    w ← padre;  η ← 1;  F(w) = f(w) + μ KL(w ‖ padre)
#    repetir hasta 5000 veces:
#        g ← (1 / Σ a) Σ_i a_i (q_i − y_i) / (q_i (1 − q_i)) p_i        con q_i = wᵀ p_i
#        brecha ← ⟨g + μ (log(w / padre) + 1), w⟩ − min_m [g + μ (log(w / padre) + 1)]_m
#        si brecha < tolerancia: parar                                   (F(w) − F* ≤ brecha)
#        repetir:                                                         (retroceso)
#            z ← (log w − η g + η μ log padre) / (1 + η μ);  w_nuevo ← exp(z) / Σ exp(z)
#            si F(w_nuevo) ≤ F(w): salir;  si no: η ← η / 2
#        si la mejora < 10⁻¹²:                                         (estancado)
#            j ← argmin_m ∇_m F(w);  t ← argmin_{t∈[0,1]} F((1 − t) w + t e_j)   (sección dorada)
#            si F((1 − t) w + t e_j) < F(w_nuevo): w_nuevo ← (1 − t) w + t e_j
#        w ← w_nuevo;  η ← 1,5 η
#    devolver w
#
# Predicción de una fila x: r ← nodo más profundo de x con pesos ajustados (si no, su ancestro);
#                           q ← w_rᵀ p(x);  p̂ ← σ(a log q − b log(1 − q) + c) si se calibró, si no q
# ```

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
