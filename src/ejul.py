"""Modelo original: Ensamble Jerárquico de Umbrales Latentes (EJUL).

Problema que resuelve
---------------------
Muchos eventos binarios de interés se definen al aplicar un umbral a una variable continua
observada después del evento: una interrupción es una semana con abastecimiento menor que el 50 %
del promedio (R < 0,5), un episodio de contaminación es un día con PM2.5 mayor que 75, una
"buena" cosecha es una calidad mayor o igual que 7. Los clasificadores estándar descartan la
variable continua y aprenden solo de la etiqueta Y = 1{R < τ}. EJUL estima la probabilidad del
evento como un funcional suavizado de la distribución condicional de R y la combina con un efecto
jerárquico de grupo estimado por Bayes empírico.

Algoritmo (entradas: X, R, τ; salida: p(x) ≈ P(R < τ | x))
-----------------------------------------------------------
1. Umbrales simétricos c_k = τ + h u_k, u_k = −1 + 2(k−1)/(K−1), k = 1..K (K impar).
2. Codificación jerárquica (opcional): para cada nivel de grupo (p. ej. corredor ⊂ municipio),
   tasa del evento con contracción Beta-Binomial hacia el nivel superior; fuera de pliegue en el
   entrenamiento. Se añaden como variables log-odds.
3. Miembros: K clasificadores probabilísticos g_k(x) ≈ F(c_k | x) = P(R < c_k | x), cada uno
   entrenado con la etiqueta 1{R < c_k}.
4. Reordenamiento monótono: el vector (g_1(x), ..., g_K(x)) se proyecta (en norma L2) sobre el
   cono de vectores no decrecientes, porque F(· | x) es una función de distribución.
5. Combinación por polinomio local de grado d en la dirección del umbral con pesos de núcleo
   κ(u_k): p̃(x) = Σ_k ω_k ĝ_k(x), ω = e_1ᵀ (Uᵀ W U)⁻¹ Uᵀ W.
6. Recalibración (opcional): regresión logística de dos parámetros sobre logit(p̃), ajustada con
   predicciones fuera de pliegue del propio entrenamiento.

Propiedades (demostradas en el capítulo 5 del libro)
---------------------------------------------------
* P1 (sesgo de segundo orden): si F(· | x) es dos veces diferenciable en τ y el diseño es simétrico
  (Σ ω_k u_k = 0), E[p̃(x)] = F(τ | x) + (h²/2) F''(τ | x) Σ ω_k u_k² + O(h³) cuando los miembros son
  insesgados; con grado 2 el término h² se anula (Σ ω_k u_k² = 0).
* P2 (el reordenamiento no empeora): la proyección sobre un conjunto convexo que contiene al
  vector verdadero es no expansiva, así que ‖Πg − F‖₂ ≤ ‖g − F‖₂.
* P3 (contracción Bayes empírica): la media posterior Beta-Binomial minimiza el riesgo cuadrático
  bayesiano y es un promedio ponderado entre la tasa del grupo y la del nivel superior, con peso
  n/(n + m) que tiende a 1 cuando el grupo tiene muchas observaciones.
* Complejidad: entrenamiento O(K · (V + 1) · T_base(n, p) + n · L), inferencia
  O(K · P_base(n, p) + n · K²), con V pliegues de recalibración y L niveles de grupo.

Solo se usan numpy y scipy para la lógica propia; los miembros son cualquier clasificador con
`predict_proba` (por defecto, XGBoost con método `hist`).
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted

NUCLEOS = ("uniforme", "epanechnikov", "triangular")


# ----------------------------------------------------------------------------------------------
# Piezas algorítmicas (funciones puras, probadas por separado en tests/)
# ----------------------------------------------------------------------------------------------
def nucleo(u: np.ndarray, tipo: str) -> np.ndarray:
    """Pesos de núcleo κ(u) en [−1, 1] (no normalizados)."""
    u = np.asarray(u, float)
    if tipo == "uniforme":
        return np.ones_like(u)
    if tipo == "epanechnikov":
        return 0.75 * (1.0 - u ** 2) + 1e-3        # el 1e-3 evita peso nulo en los extremos
    if tipo == "triangular":
        return (1.0 - np.abs(u)) + 1e-3
    raise ValueError(f"núcleo desconocido: {tipo}")


def pesos_polinomio_local(u: np.ndarray, kappa: np.ndarray, grado: int) -> np.ndarray:
    """Pesos ω del estimador de polinomio local evaluado en u = 0.

    ω = e_1ᵀ (Uᵀ W U)⁻¹ Uᵀ W, con U = [1, u, ..., u^d] y W = diag(κ). Cumplen Σ ω_k = 1 y
    Σ ω_k u_k^j = 0 para j = 1..d (reproducen exactamente polinomios de grado d).
    """
    u = np.asarray(u, float)
    if grado < 0 or grado >= len(u):
        raise ValueError("el grado debe estar entre 0 y K − 1")
    U = np.vander(u, grado + 1, increasing=True)
    W = np.diag(kappa)
    A = U.T @ W @ U
    return np.linalg.solve(A, U.T @ W)[0]


def proyeccion_isotonica(M: np.ndarray) -> np.ndarray:
    """Proyección L2 de cada fila de M sobre el cono de vectores no decrecientes.

    Usa la fórmula de mínimo-máximo de la regresión isotónica:
        ŷ_k = max_{i ≤ k} min_{j ≥ k} promedio(m_i, ..., m_j),
    vectorizada con sumas acumuladas (O(n K²) en memoria y tiempo; K es pequeño).
    """
    M = np.asarray(M, float)
    n, K = M.shape
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(M, axis=1)], axis=1)
    i_idx, j_idx = np.triu_indices(K)
    medias = np.full((n, K, K), np.nan)
    medias[:, i_idx, j_idx] = (cs[:, j_idx + 1] - cs[:, i_idx]) / (j_idx - i_idx + 1)
    salida = np.empty_like(M)
    for k in range(K):
        bloque = medias[:, : k + 1, k:]                  # i ≤ k, j ≥ k
        salida[:, k] = np.max(np.nanmin(bloque, axis=2), axis=1)
    return salida


class CodificadorJerarquico:
    """Tasa del evento por grupo con contracción Beta-Binomial jerárquica (Bayes empírico).

    Para niveles anidados de grupos (del más fino al más grueso), la tasa del nivel l es
        p_l(g) = (s_g + m_l · p_{l+1}(padre(g))) / (n_g + m_l),
    con p_{L+1} = tasa global. La fuerza del prior m_l se estima por el método de momentos
    del modelo Beta-Binomial (varianza entre grupos descontando la binomial) y se multiplica por
    `fuerza`. Grupos no vistos heredan la tasa de su padre.
    """

    def __init__(self, fuerza: float = 1.0, m_min: float = 1.0, m_max: float = 1e4):
        self.fuerza = fuerza
        self.m_min = m_min
        self.m_max = m_max

    @staticmethod
    def _momentos(s, n, p0):
        """Fuerza m del prior Beta(m p0, m (1 − p0)) por el método de momentos."""
        ok = n > 0
        tasas = s[ok] / n[ok]
        var_obs = np.average((tasas - p0) ** 2, weights=n[ok])
        var_binom = p0 * (1 - p0) * np.mean(1.0 / n[ok])
        var_entre = max(var_obs - var_binom, 1e-6)
        return p0 * (1 - p0) / var_entre - 1.0

    def fit(self, grupos: np.ndarray, y: np.ndarray):
        """`grupos`: matriz n × L de códigos enteros (−1 = sin grupo), del nivel más fino al más grueso."""
        G = np.asarray(grupos, np.int64)
        y = np.asarray(y, float)
        self.p0_ = float(y.mean())
        self.niveles_ = []
        for l in range(G.shape[1]):
            g = G[:, l]
            ok = g >= 0
            codigos, inv = np.unique(g[ok], return_inverse=True)
            s = np.bincount(inv, weights=y[ok], minlength=len(codigos))
            n = np.bincount(inv, minlength=len(codigos)).astype(float)
            m = float(np.clip(self._momentos(s, n, self.p0_) * self.fuerza, self.m_min, self.m_max))
            self.niveles_.append({"codigos": codigos, "s": s, "n": n, "m": m})
        # padres: para cada grupo del nivel l, código de su grupo en el nivel l+1 (el más frecuente)
        self.padres_ = []
        for l in range(G.shape[1] - 1):
            hijo, padre = G[:, l], G[:, l + 1]
            ok = (hijo >= 0) & (padre >= 0)
            mapa = {}
            for h, p_ in zip(hijo[ok], padre[ok]):
                mapa.setdefault(h, p_)
            self.padres_.append(mapa)
        return self

    def tasas(self, grupos: np.ndarray) -> np.ndarray:
        """Tasa contraída de cada fila en cada nivel (n × L)."""
        G = np.asarray(grupos, np.int64)
        L = len(self.niveles_)
        out = np.empty((len(G), L))
        # del nivel más grueso al más fino: la tasa del padre es el prior del hijo
        previa = np.full(len(G), self.p0_)
        for l in range(L - 1, -1, -1):
            nv = self.niveles_[l]
            pos = np.searchsorted(nv["codigos"], G[:, l])
            pos = np.clip(pos, 0, len(nv["codigos"]) - 1)
            visto = (G[:, l] >= 0) & (nv["codigos"][pos] == G[:, l])
            s = np.where(visto, nv["s"][pos], 0.0)
            n = np.where(visto, nv["n"][pos], 0.0)
            out[:, l] = (s + nv["m"] * previa) / (n + nv["m"])
            previa = out[:, l]
        return out


    def tasas_ordenadas(self, grupos: np.ndarray, y: np.ndarray, tiempos: np.ndarray) -> np.ndarray:
        """Tasas contraídas de las filas de entrenamiento usando solo filas de periodos anteriores.

        Para la fila i del periodo t_i, s_g y n_g se acumulan con las filas del mismo grupo y
        t < t_i (estadístico ordenado en el tiempo); el prior de cada nivel es la tasa ordenada del
        nivel superior y el del último nivel, la tasa global acumulada hasta t_i. Así la codificación
        de entrenamiento imita la de producción, donde solo se conoce el pasado. Las fuerzas m_l son
        las estimadas en `fit`. Implementación vectorizada con sumas acumuladas por bloques
        (grupo, periodo): O(n log n).
        """
        G = np.asarray(grupos, np.int64)
        y = np.asarray(y, float)
        _, ti = np.unique(np.asarray(tiempos), return_inverse=True)
        T = int(ti.max()) + 1
        s_t = np.bincount(ti, weights=y, minlength=T)
        n_t = np.bincount(ti, minlength=T).astype(float)
        S, N = np.cumsum(s_t) - s_t, np.cumsum(n_t) - n_t
        previa = ((S + self.p0_) / (N + 1.0))[ti]          # tasa global acumulada (prior unitario en p0)
        out = np.empty((len(G), G.shape[1]))
        for l in range(G.shape[1] - 1, -1, -1):
            g = G[:, l]
            ok = g >= 0
            clave = g[ok] * T + ti[ok]
            uk, inv = np.unique(clave, return_inverse=True)
            s_k = np.bincount(inv, weights=y[ok], minlength=len(uk))
            n_k = np.bincount(inv, minlength=len(uk)).astype(float)
            grupo = uk // T
            inicio = np.r_[True, grupo[1:] != grupo[:-1]]
            pos_inicio = np.maximum.accumulate(np.where(inicio, np.arange(len(uk)), 0))
            cs, cn = np.cumsum(s_k), np.cumsum(n_k)
            ex_s = cs - s_k - (cs - s_k)[pos_inicio]           # acumulado del grupo en periodos anteriores
            ex_n = cn - n_k - (cn - n_k)[pos_inicio]
            m = self.niveles_[l]["m"]
            col = previa.copy()
            col[ok] = (ex_s[inv] + m * previa[ok]) / (ex_n[inv] + m)
            out[:, l] = col
            previa = col
        return out


def logistica_newton(F: np.ndarray, y: np.ndarray, iteraciones: int = 100, ridge: float = 1e-6) -> np.ndarray:
    """Regresión logística P(Y=1|f) = σ(f·w + w0) por Newton-Raphson (IRLS) con un ridge mínimo.

    Devuelve (w0, w_1, ..., w_d). La log-verosimilitud es cóncava, así que Newton converge
    cuadráticamente cerca del óptimo; un paso con búsqueda lineal hacia atrás evita oscilaciones.
    """
    F = np.c_[np.ones(len(F)), np.asarray(F, float)]
    y = np.asarray(y, float)
    w = np.zeros(F.shape[1])

    def nll(w):
        z = F @ w
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * ridge * w[1:] @ w[1:]

    actual = nll(w)
    for _ in range(iteraciones):
        p = expit(F @ w)
        g = F.T @ (p - y) + ridge * np.r_[0, w[1:]]
        H = (F * (p * (1 - p))[:, None]).T @ F + ridge * np.eye(len(w))
        paso = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-4 and nll(w - t * paso) > actual + 1e-12:
            t /= 2
        w = w - t * paso
        nuevo = nll(w)
        if abs(actual - nuevo) < 1e-10 * (1 + abs(actual)):
            actual = nuevo
            break
        actual = nuevo
    return w


def rasgos_calibracion(p: np.ndarray, tipo: str) -> np.ndarray:
    """Rasgos del calibrador: logit(p) (Platt) o [ln p, −ln(1 − p)] (beta, Kull et al., 2017)."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    if tipo == "platt":
        return logit(p)[:, None]
    if tipo == "beta":
        return np.c_[np.log(p), -np.log1p(-p)]
    raise ValueError(f"calibrador desconocido: {tipo}")


def platt_newton(z: np.ndarray, y: np.ndarray, iteraciones: int = 50) -> tuple[float, float]:
    """Regresión logística de dos parámetros P(Y=1|z) = σ(a z + b) por Newton-Raphson (IRLS)."""
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    a, b = 1.0, 0.0
    for _ in range(iteraciones):
        p = expit(a * z + b)
        w = np.clip(p * (1 - p), 1e-9, None)
        g = np.array([np.sum((p - y) * z), np.sum(p - y)])
        H = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]]) + 1e-9 * np.eye(2)
        paso = np.linalg.solve(H, g)
        a, b = a - paso[0], b - paso[1]
        if np.max(np.abs(paso)) < 1e-10:
            break
    return float(a), float(b)


# ----------------------------------------------------------------------------------------------
# Estimador compatible con scikit-learn
# ----------------------------------------------------------------------------------------------
class EnsambleUmbralesLatentes(BaseEstimator, ClassifierMixin):
    """Ensamble Jerárquico de Umbrales Latentes (EJUL).

    Parameters
    ----------
    estimador_base : clasificador con `predict_proba` (se clona para cada umbral). Si es None se usa
        XGBoost (`hist`) con `params_base`.
    params_base : dict de hiperparámetros del XGBoost por defecto.
    tau : umbral que define el evento, Y = 1{R < tau}.
    h : semiancho de la banda de umbrales (en la escala de R); h = 0 equivale al clasificador base.
    n_umbrales : K, número impar de umbrales simétricos.
    nucleo : "uniforme", "epanechnikov" o "triangular".
    grado : grado del polinomio local (0 o 1 dan el promedio de núcleo; 2 elimina el sesgo h²).
    reordenar : si True, proyección isotónica de las K probabilidades antes de combinar.
    jerarquico : si True, añade la codificación Bayes empírica de los grupos.
    fuerza_eb : multiplicador de la fuerza del prior Beta estimada por momentos.
    columnas_grupo : índices (en X; se admiten negativos) de las columnas con códigos enteros de
        grupo, del nivel más fino al más grueso; esas columnas no entran a los miembros.
    calibrar : si True, recalibración con predicciones fuera de pliegue internas.
    calibrador : "platt" (σ(a·logit p + b)) o "beta" (σ(a ln p − b ln(1 − p) + c)).
    pliegues_internos : V, pliegues para la codificación y la recalibración fuera de pliegue.
    diversificar : si True, el miembro k usa la semilla random_state + k (si el estimador base la
        acepta), lo que descorrelaciona los errores de los miembros.
    codificacion : "cruzada" (ajuste cruzado aleatorio en V pliegues) o "temporal" (cada fila de
        entrenamiento se codifica solo con periodos anteriores; requiere `tiempos` en fit).
    prediccion : "completo" predice con miembros reentrenados en todos los datos; "pliegues" promedia
        las probabilidades combinadas de los V juegos de miembros de la recalibración (los mismos que
        generaron las predicciones con que se ajustó el calibrador) y no reentrena.
    random_state : semilla.

    Notes
    -----
    `fit(X, y, latente=R)`: con la variable continua R se activan los umbrales; sin ella el
    modelo se reduce al clasificador base con codificación jerárquica (K = 1).
    """

    def __init__(self, estimador_base=None, params_base=None, tau=0.5, h=0.2, n_umbrales=5,
                 nucleo="uniforme", grado=0, reordenar=True, jerarquico=True, fuerza_eb=1.0,
                 columnas_grupo=(), calibrar=True, calibrador="platt", pliegues_internos=3, diversificar=True,
                 prediccion="completo", codificacion="cruzada", random_state=42):
        self.estimador_base = estimador_base
        self.params_base = params_base
        self.tau = tau
        self.h = h
        self.n_umbrales = n_umbrales
        self.nucleo = nucleo
        self.grado = grado
        self.reordenar = reordenar
        self.jerarquico = jerarquico
        self.fuerza_eb = fuerza_eb
        self.columnas_grupo = columnas_grupo
        self.calibrar = calibrar
        self.calibrador = calibrador
        self.pliegues_internos = pliegues_internos
        self.diversificar = diversificar
        self.prediccion = prediccion
        self.codificacion = codificacion
        self.random_state = random_state

    # -------------------------------------------------------------------------------- utilidades
    def _validar_hiper(self):
        if not (isinstance(self.n_umbrales, (int, np.integer)) and self.n_umbrales >= 1 and self.n_umbrales % 2 == 1):
            raise ValueError("n_umbrales debe ser un entero impar ≥ 1")
        if self.h < 0:
            raise ValueError("h debe ser ≥ 0")
        if self.nucleo not in NUCLEOS:
            raise ValueError(f"nucleo debe ser uno de {NUCLEOS}")
        if self.grado not in (0, 1, 2):
            raise ValueError("grado debe ser 0, 1 o 2")
        if self.fuerza_eb <= 0:
            raise ValueError("fuerza_eb debe ser > 0")
        if self.codificacion not in ("cruzada", "temporal"):
            raise ValueError("codificacion debe ser 'cruzada' o 'temporal'")
        if self.prediccion not in ("completo", "pliegues"):
            raise ValueError("prediccion debe ser 'completo' o 'pliegues'")
        if self.prediccion == "pliegues" and not self.calibrar:
            raise ValueError("prediccion='pliegues' requiere calibrar=True")
        if self.calibrador not in ("platt", "beta"):
            raise ValueError("calibrador debe ser 'platt' o 'beta'")
        if self.pliegues_internos < 2:
            raise ValueError("pliegues_internos debe ser ≥ 2")

    def _nuevo_miembro(self, k: int = 0):
        if self.estimador_base is not None:
            m = clone(self.estimador_base)
        else:
            from xgboost import XGBClassifier
            params = {"n_estimators": 300, "tree_method": "hist", "n_jobs": 1, "verbosity": 0, **(self.params_base or {})}
            params.setdefault("random_state", self.random_state)
            m = XGBClassifier(**params)
        if self.diversificar and "random_state" in m.get_params():
            semilla = m.get_params()["random_state"]
            m.set_params(random_state=(self.random_state if semilla is None else semilla) + k)
        return m

    def _separar(self, X):
        # se admiten índices negativos (p. ej. (−3, −2, −1) = las tres últimas columnas)
        cg = list(dict.fromkeys(int(c) % X.shape[1] for c in self.columnas_grupo))  # conserva el orden fino → grueso
        if not cg:
            return X, None
        resto = [j for j in range(X.shape[1]) if j not in cg]
        return X[:, resto], X[:, cg].astype(np.int64)

    def _pliegues(self, n, rng):
        orden = rng.permutation(n)
        return np.array_split(orden, self.pliegues_internos)

    def _codificar(self, G, y, rng, ajustar: bool, tiempos=None):
        """Columnas log-odds de la codificación jerárquica (fuera de pliegue u ordenada si `ajustar`)."""
        if ajustar and self.codificacion == "temporal":
            self.codificador_ = CodificadorJerarquico(self.fuerza_eb).fit(G, y)
            tasas = self.codificador_.tasas_ordenadas(G, y, tiempos)
        elif ajustar:
            self.codificador_ = CodificadorJerarquico(self.fuerza_eb).fit(G, y)
            tasas = np.empty((len(G), G.shape[1]))
            for va in self._pliegues(len(G), rng):
                tr = np.setdiff1d(np.arange(len(G)), va)
                tasas[va] = CodificadorJerarquico(self.fuerza_eb).fit(G[tr], y[tr]).tasas(G[va])
        else:
            tasas = self.codificador_.tasas(G)
        return logit(np.clip(tasas, 1e-4, 1 - 1e-4))

    def _ajustar_miembros(self, Z, R, y):
        miembros = []
        for k, c in enumerate(self.umbrales_):
            etiqueta = (R < c).astype(int) if R is not None else y
            if etiqueta.min() == etiqueta.max():          # umbral sin variación: miembro constante
                miembros.append(float(etiqueta.mean()))
                continue
            m = self._nuevo_miembro(k)
            m.fit(Z, etiqueta)
            miembros.append(m)
        return miembros

    @staticmethod
    def _probas(miembros, Z):
        cols = []
        for m in miembros:
            cols.append(np.full(len(Z), m) if isinstance(m, float) else m.predict_proba(Z)[:, 1])
        return np.column_stack(cols)

    def _combinar(self, P):
        if self.reordenar and P.shape[1] > 1:
            P = proyeccion_isotonica(P)
        return np.clip(P @ self.pesos_, 1e-6, 1 - 1e-6)

    # -------------------------------------------------------------------------------- API
    def fit(self, X, y, latente=None, pliegues=None, tiempos=None):
        """Ajusta el ensamble.

        Parameters
        ----------
        X : matriz n × p (puede incluir columnas de códigos de grupo, ver `columnas_grupo`).
        y : etiquetas binarias 0/1 (Y = 1{R < tau}).
        latente : variable continua R (opcional). Si se da, debe ser coherente con y.
        pliegues : lista opcional de pares (entrenamiento, validación) de posiciones para la
            recalibración fuera de pliegue (p. ej. bloques temporales de ventana creciente). Si es
            None se usan `pliegues_internos` pliegues aleatorios. Las filas que no queden en ninguna
            validación no entran al ajuste de la recalibración.
        tiempos : periodo de cada fila (p. ej. semana objetivo); obligatorio con codificacion="temporal".
        """
        self._validar_hiper()
        X = check_array(X, ensure_all_finite="allow-nan", dtype=float)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y debe ser un vector con una etiqueta por fila de X")
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("y debe ser binaria con clases 0 y 1")
        R = None
        if latente is not None:
            R = np.asarray(latente, float)
            if R.shape != y.shape or not np.all(np.isfinite(R)):
                raise ValueError("latente debe ser un vector finito del mismo largo que y")
        rng = np.random.default_rng(self.random_state)
        K = self.n_umbrales if (R is not None and self.h > 0) else 1
        u = np.linspace(-1, 1, K) if K > 1 else np.zeros(1)
        self.u_ = u
        self.umbrales_ = self.tau + self.h * u
        self.pesos_ = pesos_polinomio_local(u, nucleo(u, self.nucleo), min(self.grado, K - 1)) if K > 1 else np.ones(1)
        Xb, G = self._separar(X)
        self.usa_grupos_ = bool(self.jerarquico and G is not None)
        if self.usa_grupos_ and self.codificacion == "temporal":
            if tiempos is None or len(tiempos) != len(y):
                raise ValueError("codificacion='temporal' requiere `tiempos` con una entrada por fila")
        Z = np.c_[Xb, self._codificar(G, y, rng, True, tiempos)] if self.usa_grupos_ else Xb
        self.n_features_in_ = X.shape[1]
        # recalibración con predicciones fuera de pliegue (los miembros de cada pliegue se descartan)
        self.w_cal_ = None
        self.oof_ = None
        if self.calibrar:
            if pliegues is None:
                todos = np.arange(len(y))
                pliegues = [(np.setdiff1d(todos, va), va) for va in self._pliegues(len(y), rng)]
            oof = np.full(len(y), np.nan)
            self.oof_miembros_ = np.full((len(y), len(self.umbrales_)), np.nan)   # sin reordenar (para la ablación)
            self.miembros_pliegue_ = []
            for tr, va in pliegues:
                tr, va = np.asarray(tr), np.asarray(va)
                miembros = self._ajustar_miembros(Z[tr], None if R is None else R[tr], y[tr])
                if self.prediccion == "pliegues":
                    self.miembros_pliegue_.append(miembros)
                P_va = self._probas(miembros, Z[va])
                self.oof_miembros_[va] = P_va
                oof[va] = self._combinar(P_va)
            ok = np.isfinite(oof)
            if ok.sum() < 10 or np.unique(y[ok]).size < 2:
                raise ValueError("los pliegues de recalibración no cubren suficientes filas de ambas clases")
            self.w_cal_ = logistica_newton(rasgos_calibracion(oof[ok], self.calibrador), y[ok])
            self.oof_crudo_ = oof
            self.oof_ = np.where(ok, self._calibrar(np.where(ok, oof, 0.5)), np.nan)
        if self.prediccion == "pliegues":
            self.miembros_ = self.miembros_pliegue_[0]       # para la API; la predicción usa todos los juegos
        else:
            self.miembros_ = self._ajustar_miembros(Z, R, y)
        return self

    def _matriz(self, X):
        check_is_fitted(self, "miembros_")
        X = check_array(X, ensure_all_finite="allow-nan", dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X tiene {X.shape[1]} columnas; se esperaban {self.n_features_in_}")
        Xb, G = self._separar(X)
        return np.c_[Xb, self._codificar(G, None, None, False)] if self.usa_grupos_ else Xb

    def probabilidades_miembros(self, X) -> np.ndarray:
        """Matriz n × K con las probabilidades de los miembros sin reordenar."""
        return self._probas(self.miembros_, self._matriz(X))

    def probabilidades_umbral(self, X) -> np.ndarray:
        """Matriz n × K con las probabilidades F̂(c_k | x) ya reordenadas (si `reordenar`)."""
        Z = self._matriz(X)
        P = self._probas(self.miembros_, Z)
        return proyeccion_isotonica(P) if (self.reordenar and P.shape[1] > 1) else P

    def _calibrar(self, p):
        if self.w_cal_ is None:
            return p
        return expit(self.w_cal_[0] + rasgos_calibracion(p, self.calibrador) @ self.w_cal_[1:])

    def probabilidad_cruda(self, X):
        """Probabilidad combinada antes de la recalibración."""
        Z = self._matriz(X)
        if self.prediccion == "pliegues":
            return np.mean([self._combinar(self._probas(m, Z)) for m in self.miembros_pliegue_], axis=0)
        return self._combinar(self._probas(self.miembros_, Z))

    def decision_function(self, X):
        """log-odds calibrados del evento."""
        return logit(np.clip(self._calibrar(self.probabilidad_cruda(X)), 1e-12, 1 - 1e-12))

    def predict_proba(self, X):
        p1 = expit(self.decision_function(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
