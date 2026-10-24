"""Modelo original: Apilamiento Jerárquico con encogimiento KL y olvido temporal (AJ-KL).

Problema que resuelve
---------------------
M modelos base (en el problema SIPSA, nueve: las mejores combinaciones de los 7 modelos del curso, LightGBM y CatBoost) estiman la probabilidad de interrupción
de cada corredor-semana. Ninguno es el mejor en todas partes: su desempeño cambia entre centrales
mayoristas y grupos de alimentos, y con el tiempo. AJ-KL combina sus probabilidades fuera de pliegue con
un pool lineal cuyos pesos dependen del nodo de una jerarquía propia del problema
(país → central → central × grupo de alimentos → corredor). Los pesos de cada nodo se encogen hacia los de su nodo
padre con una penalización de Kullback-Leibler, y las semanas antiguas pesan menos.

Formulación (nodo r con padre π(r); en la raíz, el padre es la distribución uniforme)
------------------------------------------------------------------------------------
    q_i = w_rᵀ p_i,                    p_i ∈ [ε, 1 − ε]^M,  w_r ∈ Δ^M (símplex)
    J_r(w) = Σ_{i∈r} γ^{t_max − t_i} ℓ(y_i, wᵀ p_i) + λ KL(w ‖ w_{π(r)}),
    ℓ(y, q) = −y log q − (1 − y) log(1 − q).
Al dividir entre S_r = Σ_{i∈r} γ^{t_max − t_i}: f_r(w) + μ_r KL(w ‖ w_{π(r)}), con μ_r = λ / S_r.

Regla de actualización (descenso espejo compuesto con regularizador entrópico)
---------------------------------------------------------------------------
    log w^{(k+1)} = [log w^{(k)} − η_k ∇f_r(w^{(k)}) + η_k μ_r log w_{π(r)}] / (1 + η_k μ_r) − c_k,
    ∇_m f_r(w) = (1/S_r) Σ_{i∈r} γ^{t_max − t_i} (q_i − y_i) / (q_i (1 − q_i)) p_im,
con c_k la constante que normaliza al símplex. El paso η_k se ajusta por retroceso (se acepta si J_r
baja); si se estanca, un paso de Frank-Wolfe con búsqueda lineal devuelve peso a modelos con peso ~0; la
parada usa la brecha de Frank-Wolfe ⟨∇F(w), w⟩ − min_m ∇_m F(w) < tol, que acota
F(w) − F*: los pesos devueltos son el minimizador de J_r salvo la tolerancia.

Calibración (opcional): un pool lineal de pronósticos calibrados no está calibrado salvo en casos
triviales (Ranjan y Gneiting, 2010); por eso, después del pool se ajusta una calibración beta,
σ(a ln q − b ln(1 − q) + c), por Newton-Raphson ponderado con los mismos pesos de olvido.

Propiedades (sección 5.3 del libro): convexidad estricta y unicidad del óptimo por nodo con λ > 0
(con λ = 0 el óptimo puede no ser único); casos límite (λ → 0⁺: el apilamiento local más cercano al
padre en KL; λ → ∞: todos los nodos copian la raíz, que tiende al promedio simple; profundidad 1:
apilamiento global encogido hacia la uniforme; γ = 1: sin olvido); costo O(K n M) por nivel en
entrenamiento (K iteraciones por nodo) y O(L log N + M) por fila en inferencia (búsqueda binaria en
la tabla de cada nivel).

Solo se usan numpy y scipy; la clase es compatible con scikit-learn.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted


# ----------------------------------------------------------------------------------------------
# Piezas algorítmicas
# ----------------------------------------------------------------------------------------------
def kl(w: np.ndarray, u: np.ndarray) -> float:
    """Divergencia KL(w ‖ u) entre dos puntos del símplex (con 0 log 0 = 0)."""
    w = np.asarray(w, float)
    u = np.asarray(u, float)
    m = w > 0
    return float(np.sum(w[m] * (np.log(w[m]) - np.log(u[m]))))


def objetivo_nodo(w, P, y, a, u, mu) -> float:
    """f_r(w) + μ_r KL(w ‖ u): pérdida logarítmica media ponderada más el encogimiento."""
    q = np.clip(P @ w, 1e-12, 1 - 1e-12)
    perdida = -(y * np.log(q) + (1 - y) * np.log1p(-q))
    return float(np.sum(a * perdida) / np.sum(a) + mu * kl(w, u))


def gradiente_nodo(w, P, y, a) -> np.ndarray:
    """∇f_r(w) = (1/S_r) Σ a_i (q_i − y_i) / (q_i (1 − q_i)) p_i."""
    q = np.clip(P @ w, 1e-12, 1 - 1e-12)
    r = a * (q - y) / (q * (1 - q))
    return (P.T @ r) / np.sum(a)


def brecha_fw(w, P, y, a, u, mu) -> float:
    """Brecha de Frank-Wolfe de F = f_r + μ KL(· ‖ u): ⟨∇F(w), w⟩ − min_m ∇_m F(w) ≥ 0, y = 0 solo en el óptimo.

    Como F es convexa, F(w) − F* ≤ brecha: es un certificado de optimalidad que no depende del paso.
    """
    g = gradiente_nodo(w, P, y, a)
    if mu > 0:
        g = g + mu * (np.log(np.clip(w, 1e-300, None)) - np.log(u) + 1)
    return float(g @ w - g.min())


def _paso_espejo(w, g, log_u, mu, eta):
    """Paso exacto del descenso espejo compuesto (forma cerrada en el símplex)."""
    z = (np.log(np.clip(w, 1e-300, None)) - eta * g + eta * mu * log_u) / (1 + eta * mu)
    z -= z.max()                                   # estabilidad numérica antes de exponenciar
    nuevo = np.exp(z)
    return nuevo / nuevo.sum()


def _paso_frank_wolfe(w, P, y, a, u, mu, j):
    """Paso de Frank-Wolfe hacia el vértice de menor derivada, con búsqueda lineal de sección dorada.

    El paso multiplicativo no puede dar peso a un modelo cuyo peso es ~0 (por ejemplo, porque el padre lo
    dejó en ~0); el paso hacia un vértice sí. Devuelve (w_nuevo, j_nuevo) o None si no mejora.
    """
    g = gradiente_nodo(w, P, y, a)
    if mu > 0:
        g = g + mu * (np.log(np.clip(w, 1e-300, None)) - np.log(u) + 1)
    d = -w.copy()
    d[int(np.argmin(g))] += 1.0
    F = lambda t: objetivo_nodo(np.clip(w + t * d, 0, None), P, y, a, u, mu)
    lo, hi = 0.0, 1.0
    r = (np.sqrt(5) - 1) / 2
    x1, x2 = hi - r * (hi - lo), lo + r * (hi - lo)
    f1, f2 = F(x1), F(x2)
    for _ in range(40):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - r * (hi - lo)
            f1 = F(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + r * (hi - lo)
            f2 = F(x2)
    t = (lo + hi) / 2
    nuevo = np.clip(w + t * d, 0, None)
    nuevo /= nuevo.sum()
    j_nuevo = objetivo_nodo(nuevo, P, y, a, u, mu)
    return (nuevo, j_nuevo) if j_nuevo < j else None


def descenso_espejo(P, y, a, u, mu, eta0=1.0, max_iter=5000, tol=1e-8):
    """Minimiza F(w) = f_r(w) + μ KL(w ‖ u) en el símplex por descenso espejo compuesto con paso adaptativo.

    Cada iteración aplica la actualización multiplicativa cerrada con un paso η que se ajusta por
    retroceso: se acepta el paso si F baja (si no, η se divide entre 2) y tras cada paso aceptado η crece
    un 50 %. Si el paso multiplicativo se estanca (mejora < 1e-12), se intenta un paso de Frank-Wolfe con
    búsqueda lineal, que puede devolver peso a un modelo que lo tenía en ~0. La parada usa la brecha de
    Frank-Wolfe (certificado F(w) − F* ≤ brecha < tol), no el tamaño del cambio, de modo que el resultado
    es el minimizador de J_r salvo la tolerancia.

    Returns
    -------
    (w, historial) con el valor del objetivo en cada iteración aceptada.
    """
    u = np.asarray(u, float)
    log_u = np.log(u)
    w = u.copy()                                   # inicio en el nodo padre
    j = objetivo_nodo(w, P, y, a, u, mu)
    historial = [j]
    eta = eta0
    for _ in range(max_iter):
        if brecha_fw(w, P, y, a, u, mu) < tol:
            break
        g = gradiente_nodo(w, P, y, a)
        nuevo = None
        for _ in range(60):                        # retroceso
            cand = _paso_espejo(w, g, log_u, mu, eta)
            j_cand = objetivo_nodo(cand, P, y, a, u, mu)
            if j_cand <= j:
                nuevo, j_nuevo = cand, j_cand
                break
            eta /= 2
        if nuevo is None or j - j_nuevo < 1e-12:   # estancado: paso de Frank-Wolfe
            fw = _paso_frank_wolfe(w, P, y, a, u, mu, j)
            if fw is not None and (nuevo is None or fw[1] < j_nuevo):
                nuevo, j_nuevo = fw
                eta = max(eta, eta0)
        if nuevo is None or j - j_nuevo < 1e-16:
            break                                  # ningún paso mejora: óptimo numérico
        w, j = nuevo, j_nuevo
        historial.append(j)
        eta *= 1.5
    return w, np.asarray(historial)


def logistica_newton_ponderada(F, y, peso, iteraciones=100, ridge=1e-6) -> np.ndarray:
    """Regresión logística ponderada por Newton-Raphson con búsqueda lineal; devuelve (w0, w1, ...)."""
    F = np.c_[np.ones(len(F)), np.asarray(F, float)]
    y = np.asarray(y, float)
    peso = np.asarray(peso, float) / np.mean(peso)
    w = np.zeros(F.shape[1])

    def nll(w):
        z = F @ w
        return np.sum(peso * (np.logaddexp(0, z) - y * z)) + 0.5 * ridge * w[1:] @ w[1:]

    actual = nll(w)
    for _ in range(iteraciones):
        p = expit(F @ w)
        g = F.T @ (peso * (p - y)) + ridge * np.r_[0, w[1:]]
        H = (F * (peso * p * (1 - p))[:, None]).T @ F + ridge * np.eye(len(w))
        paso = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-4 and nll(w - t * paso) > actual + 1e-12:
            t /= 2
        w = w - t * paso
        nuevo = nll(w)
        if abs(actual - nuevo) < 1e-10 * (1 + abs(actual)):
            break
        actual = nuevo
    return w


def rasgos_beta(q: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(q, float), 1e-6, 1 - 1e-6)
    return np.c_[np.log(q), -np.log1p(-q)]


# ----------------------------------------------------------------------------------------------
# Estimador compatible con scikit-learn
# ----------------------------------------------------------------------------------------------
class ApilamientoJerarquicoKL(BaseEstimator, ClassifierMixin):
    """Apilamiento jerárquico con encogimiento KL y olvido temporal (AJ-KL).

    Parameters
    ----------
    lam : λ > 0, fuerza del encogimiento hacia el nodo padre (λ = 0 desactiva el encogimiento).
    gamma : factor de olvido semanal γ ∈ (0, 1]; la fila de la semana t pesa γ^{t_max − t}.
    profundidad : niveles de la jerarquía que se usan (1 = solo la raíz, es decir, apilamiento global;
        2 = raíz y primer nivel; 3 = raíz y dos niveles; 4 = raíz y tres niveles).
    eta0 : paso inicial del descenso espejo (luego se adapta por retroceso; no es un hiperparámetro
        del modelo sino del optimizador).
    max_iter : máximo de iteraciones por nodo (salvaguarda; la parada normal es por la brecha).
    tol : tolerancia de la brecha de Frank-Wolfe (certificado de optimalidad).
    epsilon : recorte de las probabilidades de entrada a [ε, 1 − ε].
    min_filas : un nodo con menos filas no se ajusta y hereda los pesos de su padre.
    calibrar : si True, calibración beta ponderada después del pool.
    n_modelos : M, número de columnas de X con probabilidades de los modelos base; las columnas
        siguientes son códigos enteros de los niveles de la jerarquía, del más grueso al más fino
        (−1 = sin grupo). Si es None, todas las columnas son probabilidades (solo raíz).

    Notes
    -----
    `fit(X, y, tiempos=t)`: `tiempos` es el índice de la semana de cada fila (en semanas); si no se da,
    todas las filas pesan igual.
    """

    def __init__(self, lam=10.0, gamma=1.0, profundidad=3, eta0=1.0, max_iter=5000, tol=1e-8, epsilon=1e-4,
                 min_filas=1, calibrar=True, n_modelos=None):
        self.lam = lam
        self.gamma = gamma
        self.profundidad = profundidad
        self.eta0 = eta0
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
        self.min_filas = min_filas
        self.calibrar = calibrar
        self.n_modelos = n_modelos

    # ------------------------------------------------------------------------ utilidades
    def _validar_hiper(self):
        if self.lam < 0:
            raise ValueError("lam debe ser ≥ 0")
        if not (0 < self.gamma <= 1):
            raise ValueError("gamma debe estar en (0, 1]")
        if self.profundidad not in (1, 2, 3, 4):
            raise ValueError("profundidad debe ser 1, 2, 3 o 4")
        if self.eta0 <= 0 or self.max_iter < 1:
            raise ValueError("eta0 debe ser > 0 y max_iter ≥ 1")
        if not (0 < self.epsilon < 0.5):
            raise ValueError("epsilon debe estar en (0, 0,5)")

    def _separar(self, X):
        M = X.shape[1] if self.n_modelos is None else int(self.n_modelos)
        if M < 1 or M > X.shape[1]:
            raise ValueError("n_modelos no es coherente con el número de columnas de X")
        P = np.clip(X[:, :M], self.epsilon, 1 - self.epsilon)
        if np.any((X[:, :M] < 0) | (X[:, :M] > 1)):
            raise ValueError("las columnas de probabilidades deben estar en [0, 1]")
        G = X[:, M:].astype(np.int64) if X.shape[1] > M else np.empty((len(X), 0), np.int64)
        return P, G

    # ------------------------------------------------------------------------ API
    def fit(self, X, y, tiempos=None):
        self._validar_hiper()
        X = check_array(X, dtype=float)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y debe tener una etiqueta por fila de X")
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("y debe ser binaria con clases 0 y 1")
        y = y.astype(float)
        P, G = self._separar(X)
        self.n_features_in_ = X.shape[1]
        self.M_ = P.shape[1]
        niveles = min(self.profundidad - 1, G.shape[1])
        if tiempos is None:
            a = np.ones(len(y))
        else:
            t = np.asarray(tiempos, float)
            if t.shape != y.shape or not np.all(np.isfinite(t)):
                raise ValueError("tiempos debe ser un vector finito con una entrada por fila")
            a = self.gamma ** (t.max() - t)
        self.peso_efectivo_ = float(a.sum())
        uniforme = np.full(self.M_, 1.0 / self.M_)
        # raíz
        mu = self.lam / a.sum()
        w_raiz, hist = descenso_espejo(P, y, a, uniforme, mu, self.eta0, self.max_iter, self.tol)
        self.raiz_ = w_raiz
        self.historial_ = {("raiz",): hist}
        self.pesos_ = [dict() for _ in range(niveles)]
        self.n_nodo_ = [dict() for _ in range(niveles)]
        # niveles inferiores, de arriba hacia abajo; el padre de un nodo del nivel l es el nodo del
        # nivel l − 1 que contiene a sus filas (la jerarquía es anidada)
        for l in range(niveles):
            codigos = G[:, l]
            # agrupación por código en O(n log n): un solo ordenamiento y cortes, en vez de una máscara por nodo
            orden = np.argsort(codigos, kind="stable")
            ordenados = codigos[orden]
            unicos, inicios = np.unique(ordenados, return_index=True)
            fines = np.r_[inicios[1:], len(ordenados)]
            for c, ini, fin in zip(unicos, inicios, fines):
                if c < 0 or fin - ini < self.min_filas:
                    continue
                idx = orden[ini:fin]
                if l == 0:
                    padre = self.raiz_
                else:
                    padre = self.pesos_[l - 1].get(G[idx[0], l - 1], self.raiz_)
                mu = self.lam / a[idx].sum()
                w, hist = descenso_espejo(P[idx], y[idx], a[idx], padre, mu, self.eta0, self.max_iter, self.tol)
                self.pesos_[l][int(c)] = w
                self.n_nodo_[l][int(c)] = len(idx)
                self.historial_[(l, int(c))] = hist
        # calibración beta ponderada sobre el pool
        self.w_cal_ = None
        if self.calibrar:
            q = self._pool(P, G)
            self.w_cal_ = logistica_newton_ponderada(rasgos_beta(q), y, a)
        return self

    def pesos_por_fila(self, X) -> np.ndarray:
        """Matriz n × M con los pesos del nodo más profundo disponible para cada fila."""
        check_is_fitted(self, "raiz_")
        X = check_array(X, dtype=float)
        _, G = self._separar(X)
        return self._pesos(G, len(X))

    def _pesos(self, G, n):
        """Pesos del nodo más profundo disponible para cada fila, con tablas indexadas por código: O(n M)."""
        W = np.tile(self.raiz_, (n, 1))
        for l, tabla in enumerate(self.pesos_):
            if l >= G.shape[1] or not tabla:
                continue
            codigos = G[:, l]
            claves = np.fromiter(tabla.keys(), dtype=np.int64)
            orden = np.argsort(claves)
            claves = claves[orden]
            valores = np.vstack([tabla[int(k)] for k in claves])
            pos = np.searchsorted(claves, codigos)
            pos = np.clip(pos, 0, len(claves) - 1)
            visto = (codigos >= 0) & (claves[pos] == codigos)
            W[visto] = valores[pos[visto]]
        return W

    def _pool(self, P, G):
        W = self._pesos(G, len(P))
        return np.clip(np.sum(W * P, axis=1), 1e-6, 1 - 1e-6)

    def probabilidad_pool(self, X) -> np.ndarray:
        """Probabilidad del pool lineal antes de la calibración."""
        check_is_fitted(self, "raiz_")
        X = check_array(X, dtype=float)
        P, G = self._separar(X)
        return self._pool(P, G)

    def decision_function(self, X):
        check_is_fitted(self, "raiz_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X tiene {X.shape[1]} columnas; se esperaban {self.n_features_in_}")
        P, G = self._separar(X)
        q = self._pool(P, G)
        if self.w_cal_ is not None:
            return self.w_cal_[0] + rasgos_beta(q) @ self.w_cal_[1:]
        return logit(q)

    def predict_proba(self, X):
        p1 = expit(self.decision_function(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
