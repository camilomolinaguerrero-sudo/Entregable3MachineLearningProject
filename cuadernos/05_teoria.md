# 5. Modelo original: apilamiento jerárquico con encogimiento KL y olvido temporal (AJ-KL)

## 5.1 Motivación

Los modelos del Entregable 2 quedan a menos de 0,02 de AUC entre sí y XGBoost es el mejor en promedio,
pero no en todas partes. Su ventaja cambia entre centrales mayoristas, grupos de alimentos y corredores, y
cambia con el tiempo: un modelo puede ser el mejor para las frutas que llegan a una central de la costa y no
para los tubérculos que llegan a Bogotá. Un único vector de pesos (el apilamiento clásico) ignora esa
heterogeneidad; estimar un vector de pesos independiente por corredor la sobreajusta, porque muchos
corredores tienen pocas semanas.

El AJ-KL toma una posición intermedia con una estructura que ya tiene el problema. Combina las
probabilidades de los modelos base con pesos que dependen del nodo de una jerarquía
(país → central → central × grupo de alimentos → corredor). Los pesos de cada nodo se encogen hacia los de
su nodo padre con una penalización de Kullback-Leibler, de modo que un corredor con pocos datos hereda los
pesos de su central y uno con muchos datos se aparta de ellos solo si la evidencia lo justifica, y las semanas
antiguas pesan menos que las recientes. La propuesta sale de la revisión de literatura del capítulo 4, que la
ubica en la familia de los ensambles con regla de ponderación propia que la guía admite como contribución
original (cuadro 1).

## 5.2 Formulación

Sean $i = 1, \dots, n$ las corredor-semanas, $t_i$ su semana, $y_i \in \{0, 1\}$ la interrupción y
$p_i = (p_{i1}, \dots, p_{iM}) \in [\varepsilon, 1 - \varepsilon]^M$ las probabilidades fuera de muestra de
los $M$ modelos base. La jerarquía $\mathcal{T}$ tiene raíz $\rho$; cada nodo $r \neq \rho$ tiene un padre
$\pi(r)$ y un vector de pesos $w_r$ en el símplex
$\Delta^M = \{w \in \mathbb{R}^M : w_m \ge 0, \sum_m w_m = 1\}$. La predicción para una fila de la hoja $r$
es un pool lineal

$$
q_i = w_r^\top p_i .
$$ (eq-pool)

Los pesos de cada nodo se estiman, de la raíz hacia las hojas, como

$$
w_r = \arg\min_{w \in \Delta^M} J_r(w),\qquad
J_r(w) = \sum_{i \in r} \gamma^{\,t_{\max} - t_i}\, \ell(y_i, w^\top p_i) + \lambda\, \mathrm{KL}(w \,\|\, w_{\pi(r)}),
$$ (eq-objetivo-ajkl)

con la pérdida logarítmica $\ell(y, q) = -y \log q - (1 - y)\log(1 - q)$, el factor de olvido
$\gamma \in (0, 1]$, la fuerza del encogimiento $\lambda \ge 0$ y, en la raíz, $w_{\pi(\rho)} = u$ la
distribución uniforme. Al dividir por $S_r = \sum_{i \in r}\gamma^{t_{\max} - t_i}$ (el tamaño de muestra
efectivo del nodo) el problema queda $f_r(w) + \mu_r \mathrm{KL}(w \| w_{\pi(r)})$ con $f_r$ la pérdida media
ponderada y $\mu_r = \lambda / S_r$: el encogimiento efectivo crece cuando el nodo tiene pocos datos.

**Regla de actualización.** Cada nodo se resuelve con descenso espejo compuesto con el regularizador
entrópico {cite}`beck2003mirror,duchi2010composite`. En la iteración $k$, con paso $\eta_k$,

$$
w^{(k+1)} = \arg\min_{w \in \Delta^M}\; \eta_k \langle \nabla f_r(w^{(k)}), w\rangle + \eta_k \mu_r \mathrm{KL}(w \| w_{\pi(r)}) + \mathrm{KL}(w \| w^{(k)}),
$$ (eq-comid)

que tiene solución cerrada (proposición 2):

$$
\log w^{(k+1)}_m = \frac{\log w^{(k)}_m - \eta_k \nabla_m f_r(w^{(k)}) + \eta_k \mu_r \log w_{\pi(r),m}}{1 + \eta_k \mu_r} - c_k,
\qquad
\nabla_m f_r(w) = \frac{1}{S_r}\sum_{i \in r} \gamma^{t_{\max} - t_i}\, \frac{q_i - y_i}{q_i(1 - q_i)}\, p_{im},
$$ (eq-actualizacion)

donde $c_k$ normaliza al símplex. El algoritmo empieza en los pesos del padre. El paso se ajusta por
retroceso: se acepta si $F_r = f_r + \mu_r\mathrm{KL}(\cdot\|w_{\pi(r)})$ baja (si no, $\eta$ se divide
entre 2) y después de cada paso aceptado crece un 50 %. La actualización multiplicativa tiene un punto
ciego: si un peso está en casi cero (por ejemplo, porque el padre casi no usa ese modelo), solo puede crecer
en proporción a sí mismo y el algoritmo se estanca lejos del óptimo. Por eso, cuando el paso multiplicativo
mejora $F_r$ en menos de $10^{-12}$, se prueba un paso de Frank-Wolfe {cite}`jaggi2013revisiting`:
$w \leftarrow (1 - t)\,w + t\, e_j$, con $j = \arg\min_m \nabla_m F_r(w)$ y $t \in [0, 1]$ elegido por
búsqueda lineal de sección dorada, que sí puede devolverle peso a ese modelo. La parada no usa el tamaño del cambio sino la brecha
de Frank-Wolfe {cite}`jaggi2013revisiting`,

$$
g(w) = \langle \nabla F_r(w), w\rangle - \min_m \nabla_m F_r(w),
$$ (eq-brecha)

que acota el error de optimización (proposición 3): el algoritmo se detiene cuando $g(w) < 10^{-8}$ o
cuando ningún paso reduce $F_r$, de modo que los pesos devueltos son el minimizador de $J_r$ salvo la
tolerancia numérica.

*Versión 2.0 y corrección.* La primera implementación usaba $\eta_k = \eta_0/\sqrt{k}$ con $K \le 300$
iteraciones y paraba por $\|w^{(k+1)} - w^{(k)}\|_1$. La revisión adversarial del código (capítulo 8)
mostró que con nueve modelos base muy correlacionados el objetivo es casi plano y esos pasos dejaban los
pesos de la raíz lejos del óptimo ($\|w - w^\star\|_1 = 0{,}53$ en k1): $\eta_0$ y $K$ funcionaban como
una regularización implícita hacia el padre que la teoría no describía. La versión 2.1, que es la que se
reporta, resuelve cada nodo hasta el certificado y llega al mismo óptimo que L-BFGS en 106 iteraciones
(0,05 s) en ese caso; $\eta_0$ y $K$ dejaron de ser hiperparámetros. El paso de Frank-Wolfe se añadió
después de revisar las brechas por nodo (sección 5.7): sin él, algunos nodos de central cuyo padre daba peso
casi nulo a un modelo útil se detenían con brechas de hasta $3 \times 10^{-3}$. Una fila
cuyo nodo no tiene pesos (no se vio en el entrenamiento) usa los de su ancestro más profundo que sí los
tenga.

**Calibración.** Un pool lineal de pronósticos calibrados no está calibrado salvo que todos coincidan
{cite}`ranjan2010combining`: queda subdisperso, con probabilidades encogidas hacia el centro. Por eso,
después del pool se ajusta una calibración beta $\sigma(a \log q - b\log(1 - q) + c)$ {cite}`kull2017beta`
por Newton-Raphson ponderado con los mismos pesos de olvido. Es una transformación monótona: no cambia el
AUC y se activa solo si mejora la log-loss en la validación interna.

## 5.3 Propiedades

**Proposición 1 (convexidad estricta y unicidad).** Para $\lambda > 0$ y $p_i \in [\varepsilon, 1-\varepsilon]^M$,
$J_r$ es estrictamente convexa en $\Delta^M$ y tiene un único minimizador; si además el padre tiene
soporte completo ($w_{\pi(r),m} > 0$ para todo $m$), el minimizador está en el interior del símplex.

*Demostración.* $q = w^\top p$ es afín en $w$ y $-\log q$ y $-\log(1 - q)$ son convexas en $q \in (0,1)$, así
que cada término $\ell(y_i, w^\top p_i)$ es convexo en $w$ y la suma ponderada con pesos positivos también.
$\mathrm{KL}(w \| v) = \sum_m w_m \log(w_m / v_m)$ es estrictamente convexa en $w$ (su hessiano es
$\mathrm{diag}(1/w_m) \succ 0$). La suma de una función convexa y una estrictamente convexa es
estrictamente convexa; $\Delta^M$ es compacto y convexo y $J_r$ es continua, así que existe un minimizador y
es único. Como $\partial \mathrm{KL}/\partial w_m = \log(w_m/v_m) + 1 \to -\infty$ cuando $w_m \to 0$ y el
gradiente de la pérdida está acotado (proposición 3), el óptimo no puede tener componentes nulas. $\square$

Con $\lambda = 0$ la convexidad se conserva pero la unicidad no está garantizada: $f_r$ es estrictamente
convexa en $q = P_r w$, no en $w$, y si dos modelos base son casi colineales en el nodo el conjunto de
minimizadores es un politopo. Por eso la ablación «λ = 0» se interpreta como un caso límite (proposición 4).

**Proposición 2 (la actualización es exacta y sus puntos fijos son las condiciones KKT).** La expresión
{eq}`eq-actualizacion` es la solución única del subproblema {eq}`eq-comid`, y $w^\star$ es punto fijo de la
iteración si y solo si cumple las condiciones de Karush-Kuhn-Tucker del problema del nodo.

*Demostración.* El lagrangiano del subproblema, con multiplicador $\nu$ para $\sum_m w_m = 1$, es
$\eta \langle g, w\rangle + \eta\mu \sum_m w_m \log(w_m/v_m) + \sum_m w_m \log(w_m / w^{(k)}_m) + \nu(\sum_m w_m - 1)$,
con $g = \nabla f(w^{(k)})$ y $v = w_{\pi(r)}$. La entropía mantiene $w_m > 0$, así que las restricciones de
no negatividad no se activan. Igualando a cero la derivada respecto de $w_m$:
$\eta g_m + \eta\mu(\log w_m - \log v_m + 1) + \log w_m - \log w^{(k)}_m + 1 + \nu = 0$, de donde
$(1 + \eta\mu)\log w_m = \log w^{(k)}_m - \eta g_m + \eta\mu \log v_m - \text{cte}$, que es
{eq}`eq-actualizacion` con $c_k$ fijado por la normalización. El subproblema es estrictamente convexo, así
que la solución es única. Si $w^{(k+1)} = w^{(k)} = w^\star$, la ecuación queda
$\nabla_m f(w^\star) + \mu(\log w^\star_m - \log v_m + 1) = $ cte para todo $m$, que es exactamente la
condición de estacionariedad de $f + \mu\mathrm{KL}(\cdot\|v)$ sobre el símplex con multiplicador común; por
convexidad (proposición 1) ese punto es el óptimo. $\square$

**Proposición 3 (certificado de optimalidad y convergencia).** (a) Para todo $w \in \Delta^M$,
$F_r(w) - F_r^\star \le g(w)$, con $g$ la brecha {eq}`eq-brecha`. (b) Con $p_i \in [\varepsilon, 1-\varepsilon]^M$,
$f_r$ es $L$-suave respecto de $\|\cdot\|_1$ con $L \le 1/\varepsilon^2$, y el descenso espejo compuesto con
paso constante $\eta \le 1/L$ cumple

$$
F_r(w^{(K)}) - F_r^\star \le \frac{\mathrm{KL}(w_r^\star \,\|\, w_{\pi(r)})}{\eta K}.
$$ (eq-tasa)

*Demostración.* (a) Por convexidad, $F_r(w) - F_r(w^\star) \le \langle \nabla F_r(w), w - w^\star\rangle$, y como
$w^\star \in \Delta^M$, $\langle \nabla F_r(w), w^\star\rangle \ge \min_m \nabla_m F_r(w)$; restando queda la
brecha. (b) El hessiano de $f_r$ es $\frac{1}{S_r}\sum_i a_i \kappa_i\, p_i p_i^\top$ con
$\kappa_i = y_i/q_i^2 + (1-y_i)/(1-q_i)^2 \le 1/\varepsilon^2$, porque $q_i$ es una combinación convexa de
valores en $[\varepsilon, 1-\varepsilon]$; para $d \in \mathbb{R}^M$, $(p_i^\top d)^2 \le \|d\|_1^2$ porque
$p_{im} \le 1$, así que $d^\top \nabla^2 f_r\, d \le \|d\|_1^2/\varepsilon^2$. La entropía negativa es
1-fuertemente convexa respecto de $\|\cdot\|_1$ en el símplex (desigualdad de Pinsker), de modo que $f_r$ es
$L$-suave relativa a ella. El término $\mu_r\mathrm{KL}(\cdot\|w_{\pi(r)})$ se trata de forma exacta en el
paso (proposición 2), y el resultado de convergencia del método de gradiente de Bregman para funciones
relativamente suaves {cite}`lu2018relatively` (teorema 3.1, que se extiende al término compuesto como en
{cite:t}`duchi2010composite`) da {eq}`eq-tasa`, con la divergencia inicial medida desde el punto de partida,
que es el padre. $\square$

La cota de (b) es solo una garantía en el peor caso: con $\varepsilon = 10^{-4}$, $1/L$ es diminuto. Por eso
la implementación adapta el paso por retroceso y usa (a) como criterio de parada: no importa cuántas
iteraciones tome, al detenerse con $g(w) < 10^{-8}$ el error en el objetivo es menor que $10^{-8}$. Con $\mu_r > 0$
el objetivo es además $\mu_r$-fuertemente convexo relativo a la entropía, de modo que
$\mathrm{KL}(w^\star\|w) \le g(w)/\mu_r$ también acota la distancia a los pesos óptimos. La sección 5.7 reporta
las brechas alcanzadas en el modelo final.

**Proposición 4 (casos límite).** (a) Con $\lambda \to 0^+$ cada nodo tiende a un apilamiento local sin
encogimiento: entre los minimizadores de $f_r$, el más cercano a los pesos del padre en KL.
(b) Con $\lambda \to \infty$, todos los pesos convergen a la uniforme: el modelo se vuelve el promedio simple.
(c) Con profundidad 1 (solo la raíz) el modelo es un apilamiento global en el símplex encogido hacia la
uniforme con $\mu_\rho = \lambda / S_\rho$, que es despreciable cuando $S_\rho \gg \lambda$. (d) Con
$\gamma = 1$ desaparece el olvido.

*Demostración.* (d) es inmediata de {eq}`eq-objetivo-ajkl`. (a) $f_r$ es convexa y continua en el compacto
$\Delta^M$ y $\mathrm{KL}(\cdot\|w_{\pi(r)})$ es estrictamente convexa y continua; el resultado clásico de
regularización de Tikhonov da que los minimizadores de $f_r + \mu\mathrm{KL}$ convergen, cuando $\mu \to 0^+$,
al elemento de $\arg\min f_r$ que minimiza $\mathrm{KL}(\cdot\|w_{\pi(r)})$. (b) Dividiendo $J_r$ por $\lambda$, el
problema converge a $\min_w \mathrm{KL}(w \| w_{\pi(r)})$, cuyo único mínimo es $w = w_{\pi(r)}$; por
inducción desde la raíz (cuyo padre es $u$), todos los nodos tienden a $u$. (c) El término de encogimiento de
la raíz pesa $\mu_\rho = \lambda / S_\rho \to 0$ cuando $S_\rho$ crece, así que la raíz minimiza la
log-loss ponderada del pool en el símplex, que es la definición del apilamiento lineal global. $\square$

Las reglas estándar contra las que se contrasta el modelo (promedio simple, apilamiento global y pesos por
error global) son, por tanto, casos particulares o vecinos del AJ-KL, lo que hace natural la ablación.

**Proposición 5 (interpretación bayesiana del encogimiento).** El problema del nodo es el estimador de máxima
a posteriori bajo una verosimilitud de Bernoulli con exponente $\gamma^{t_{\max}-t_i}$ (verosimilitud
ponderada) y el prior entrópico $\pi(w) \propto \exp\{-\lambda \mathrm{KL}(w \| w_{\pi(r)})\}$ centrado en los
pesos del padre.

*Demostración.* El logaritmo del posterior es
$-\sum_i \gamma^{t_{\max}-t_i}\ell(y_i, w^\top p_i) - \lambda \mathrm{KL}(w\|w_{\pi(r)}) + $ cte $= -J_r(w) +$ cte. $\square$

Como en el Bayes empírico jerárquico, el peso del prior frente a los datos es $\lambda$ frente a $S_r$: con
$S_r \gg \lambda$ dominan los datos del nodo; con $S_r \ll \lambda$ el nodo copia a su padre. El olvido reduce
$S_r$ (el tamaño de muestra efectivo con semivida $\log 2 / \log(1/\gamma)$ semanas), así que los nodos con
datos antiguos se encogen más: el modelo confía menos en evidencia vieja.

**Proposición 6 (capacidad y generalización).** Fijados los modelos base, la clase de predictores del nodo
$\{x \mapsto w^\top p(x) : w \in \Delta^M\}$ es la envolvente convexa de $M$ funciones con valores en
$[0, 1]$, y su complejidad de Rademacher empírica con $n_r$ filas es a lo sumo $\sqrt{2\log M / n_r}$.

*Demostración.* El supremo de una función lineal sobre el símplex se alcanza en un vértice, así que la
complejidad de Rademacher de la envolvente convexa es igual a la del conjunto finito de los $M$ modelos base;
el lema de Massart para clases finitas con valores en $[0,1]$ da $\sqrt{2 \log M / n_r}$. $\square$

La cota crece con el logaritmo del número de modelos y no con el número de variables: el combinador tiene muy
poca capacidad por nodo, y la jerarquía con encogimiento evita que los nodos pequeños (con $n_r$ chico y
cota grande) se aparten mucho de su padre. Esa es la justificación de usar nodos tan finos como el corredor.

## 5.4 Complejidad

Sea $L$ el número de niveles, $K$ el número de iteraciones por nodo (incluidos los retrocesos) y $n$ el
número de filas del historial. En cada nivel, cada fila pertenece a un solo nodo, así que una iteración sobre
todos los nodos del nivel cuesta $O(nM)$ (evaluar el pool, el objetivo y el gradiente); agrupar las filas por
nodo cuesta $O(n\log n)$ (un ordenamiento por nivel). El entrenamiento cuesta $O(L\,(n \log n + K\,n\,M))$
más el costo de los modelos base, que ya están entrenados. La inferencia cuesta $O(L \log N + M)$ por fila,
con $N$ el número de nodos del nivel (búsqueda binaria del código en la tabla de cada nivel y un producto
interno de longitud $M$); la memoria es $O(|\mathcal{T}|\, M)$. En el modelo final $K$ fue a lo sumo del
orden de cien iteraciones por nodo (sección 5.7).
La sección 5.7 contrasta estas expresiones con tiempos medidos.

## 5.5 Pseudocódigo

```text
Entrada: probabilidades fuera de muestra P (n × M) de los modelos base, etiquetas y, semanas t,
         códigos de la jerarquía por nivel (central, central × grupo, corredor),
         hiperparámetros λ, γ, profundidad; tolerancia (10⁻⁸), ε; calibrar ∈ {sí, no}
Salida:  pesos w_r por nodo; calibrador beta (opcional)

1. P ← recortar(P, ε, 1 − ε);  a_i ← γ^(t_max − t_i)
2. w_raíz ← DescensoEspejo(P, y, a, padre = uniforme, μ = λ / Σ a_i)
3. para cada nivel l = 1 .. profundidad − 1:
       para cada nodo r del nivel l:
           w_r ← DescensoEspejo(P_r, y_r, a_r, padre = w_{π(r)}, μ = λ / Σ_{i∈r} a_i)
4. si calibrar: (a, b, c) ← Newton ponderado de y sobre [log q, −log(1 − q)] con pesos a,
   donde q_i = w_{nodo(i)}ᵀ p_i
5. devolver {w_r}, (a, b, c)

DescensoEspejo(P, y, a, padre, μ):
   w ← padre;  η ← 1;  F(w) = f(w) + μ KL(w ‖ padre)
   repetir hasta 5000 veces:
       g ← (1 / Σ a) Σ_i a_i (q_i − y_i) / (q_i (1 − q_i)) p_i        con q_i = wᵀ p_i
       brecha ← ⟨g + μ (log(w / padre) + 1), w⟩ − min_m [g + μ (log(w / padre) + 1)]_m
       si brecha < tolerancia: parar                                   (F(w) − F* ≤ brecha)
       repetir:                                                         (retroceso)
           z ← (log w − η g + η μ log padre) / (1 + η μ);  w_nuevo ← exp(z) / Σ exp(z)
           si F(w_nuevo) ≤ F(w): salir;  si no: η ← η / 2
       si la mejora < 10⁻¹²:                                         (estancado)
           j ← argmin_m ∇_m F(w);  t ← argmin_{t∈[0,1]} F((1 − t) w + t e_j)   (sección dorada)
           si F((1 − t) w + t e_j) < F(w_nuevo): w_nuevo ← (1 − t) w + t e_j
       w ← w_nuevo;  η ← 1,5 η
   devolver w

Predicción de una fila x: r ← nodo más profundo de x con pesos ajustados (si no, su ancestro);
                          q ← w_rᵀ p(x);  p̂ ← σ(a log q − b log(1 − q) + c) si se calibró, si no q
```
