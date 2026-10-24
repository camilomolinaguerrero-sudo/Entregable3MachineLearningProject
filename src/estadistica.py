"""Comparación estadística de modelos (sección 6 de la guía) y pruebas sobre residuos.

Clasificación: Friedman → Nemenyi (diagrama de diferencia crítica) → DeLong con Holm → delta de
Cliff, con intervalos bootstrap BCa por bloques de semana.

Regresión: Model Confidence Set → Giacomini-White / Clark-West → Diebold-Mariano con corrección
HLN e intervalo por bootstrap estacionario → d de Cohen; además Reality Check / SPA de Hansen.

Residuos: White (heterocedasticidad), BDS (independencia no lineal), Ljung-Box y ACF, normalidad.

La implementación rápida de DeLong sigue a Sun y Xu (2014), adaptada de la Tarea 1 (validada allí
contra la versión directa O(mn) y contra pROC).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

Z975 = stats.norm.ppf(0.975)


# ----------------------------------------------------------------------------------------------
# DeLong
# ----------------------------------------------------------------------------------------------
def _delong_rapido(puntajes: np.ndarray, m: int):
    """Algoritmo de Sun y Xu (2014). `puntajes`: k × (m + n), positivos en las primeras m columnas."""
    n = puntajes.shape[1] - m
    k = puntajes.shape[0]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r in range(k):
        tx[r] = stats.rankdata(puntajes[r, :m])
        ty[r] = stats.rankdata(puntajes[r, m:])
        tz[r] = stats.rankdata(puntajes[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n


def delong_prueba(y, s1, s2) -> dict:
    """Prueba de DeLong para H0: AUC1 = AUC2 con las mismas observaciones (pareada)."""
    y = np.asarray(y).astype(int)
    orden = np.r_[np.flatnonzero(y == 1), np.flatnonzero(y == 0)]
    P = np.vstack([np.asarray(s1, float)[orden], np.asarray(s2, float)[orden]])
    auc, cov = _delong_rapido(P, int(y.sum()))
    delta = auc[0] - auc[1]
    se = np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-300))
    z = delta / se
    return {"auc1": auc[0], "auc2": auc[1], "delta": delta, "delta_inf": delta - Z975 * se,
            "delta_sup": delta + Z975 * se, "z": z, "p": 2 * stats.norm.sf(abs(z)),
            "correlacion": cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])}


def ajustar_p(p, metodo: str = "holm") -> np.ndarray:
    """Holm-Bonferroni ("holm") o Benjamini-Hochberg ("fdr_bh")."""
    return multipletests(np.asarray(p, float), alpha=0.05, method=metodo)[1]


def escala_dauc(d: float) -> str:
    """Tabla de interpretación de ΔAUC del curso (sección 9.9.1)."""
    d = abs(d)
    if d < 0.01:
        return "trivial"
    if d < 0.03:
        return "pequeña"
    if d < 0.05:
        return "moderada"
    return "importante"


# ----------------------------------------------------------------------------------------------
# Friedman, Nemenyi y diagrama de diferencia crítica
# ----------------------------------------------------------------------------------------------
def friedman(tabla: pd.DataFrame, mayor_mejor: bool = True) -> dict:
    """Prueba de Friedman sobre una tabla bloques × tratamientos (filas = bloques).

    Devuelve el estadístico chi-cuadrado, su p, la corrección F de Iman-Davenport (menos
    conservadora) y los rangos promedio (1 = mejor).
    """
    X = tabla.to_numpy(float)
    N, k = X.shape
    rangos = np.apply_along_axis(lambda r: stats.rankdata(-r if mayor_mejor else r), 1, X)
    R = rangos.mean(axis=0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    # corrección por empates: C = 1 - sum(t^3 - t) / (N k (k^2 - 1))
    empates = 0.0
    for fila in X:
        _, t = np.unique(fila, return_counts=True)
        empates += np.sum(t ** 3 - t)
    C = 1 - empates / (N * k * (k ** 2 - 1))
    chi2 = chi2 / C if C > 0 else chi2
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) != chi2 else np.inf
    return {"chi2": chi2, "gl": k - 1, "p": stats.chi2.sf(chi2, k - 1), "F_ID": ff,
            "gl_F": (k - 1, (k - 1) * (N - 1)), "p_F": stats.f.sf(ff, k - 1, (k - 1) * (N - 1)),
            "N": N, "k": k, "rangos": pd.Series(R, index=tabla.columns).sort_values()}


def friedman_permutacion(tabla: pd.DataFrame, grupos, mayor_mejor: bool = True, B: int = 10000,
                         semilla: int = 42) -> dict:
    """p-valor de Friedman por permutación restringida a grupos de bloques dependientes.

    Los bloques de un mismo grupo (p. ej., todas las combinaciones evaluadas en el mismo pliegue
    externo) comparten datos y año, así que no son réplicas independientes. Bajo H0 se permutan
    las etiquetas de los tratamientos con la MISMA permutación en todos los bloques de un grupo
    (la unidad intercambiable es el grupo), y se recalcula el chi-cuadrado de Friedman. Así, una
    interacción tratamiento × grupo (un optimizador con "suerte" en un año) no se confunde con un
    efecto del tratamiento.
    """
    X = tabla.to_numpy(float)
    N, k = X.shape
    rangos = np.apply_along_axis(lambda r: stats.rankdata(-r if mayor_mejor else r), 1, X)
    grupos = np.asarray(grupos)
    unicos = np.unique(grupos)
    idx = [np.flatnonzero(grupos == g) for g in unicos]
    sumas = np.stack([rangos[i].sum(axis=0) for i in idx])          # grupos × k

    def estad(S):
        R = S.sum(axis=0) / N
        return 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)

    obs = estad(sumas)
    rng = np.random.default_rng(semilla)
    sim = np.empty(B)
    for b in range(B):
        S = np.stack([fila[rng.permutation(k)] for fila in sumas])
        sim[b] = estad(S)
    return {"chi2": obs, "p_perm": (np.sum(sim >= obs - 1e-12) + 1) / (B + 1), "grupos": len(unicos), "B": B}


def dominancia_pareada(a, b) -> tuple[float, str]:
    """Delta de Cliff para muestras pareadas: media de sign(a_i - b_i) (dominancia de las
    diferencias), con la escala de magnitud de Romano et al. (2006)."""
    d = float(np.mean(np.sign(np.asarray(a, float) - np.asarray(b, float))))
    x = abs(d)
    mag = "despreciable" if x < 0.147 else "pequeño" if x < 0.33 else "mediano" if x < 0.474 else "grande"
    return d, mag


def diferencia_critica(k: int, N: int, alfa: float = 0.05) -> float:
    """CD de Nemenyi: q_alfa sqrt(k(k+1)/(6N)), con q_alfa del rango studentizado / sqrt(2)."""
    q = stats.studentized_range.ppf(1 - alfa, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6 * N)))


def nemenyi(tabla: pd.DataFrame, mayor_mejor: bool = True) -> pd.DataFrame:
    """p-valores de Nemenyi por pares (scikit-posthocs) sobre la misma tabla de Friedman."""
    import scikit_posthocs as sp
    datos = -tabla if mayor_mejor else tabla
    return sp.posthoc_nemenyi_friedman(datos.to_numpy(float)).set_axis(tabla.columns, axis=0).set_axis(tabla.columns, axis=1)


def grafico_cd(rangos: pd.Series, cd: float, ax, titulo: str = "", max_etiqueta: int = 34):
    """Diagrama de diferencia crítica al estilo de Demšar (2006).

    Los modelos se ubican por rango promedio (izquierda = mejor). Las barras horizontales
    gruesas unen grupos de modelos cuya diferencia de rangos es menor que la CD (no
    significativamente distintos según Nemenyi).
    """
    import matplotlib.pyplot as plt  # noqa: F401
    r = rangos.sort_values()
    k = len(r)
    lo, hi = 1, max(k, int(np.ceil(r.max())))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    alto = max(3, k * 0.5)
    ax.set_ylim(alto, -1.5)
    ax.hlines(0, lo, hi, color="k", lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, -0.12, 0.12, color="k", lw=1)
        ax.text(t, -0.35, str(t), ha="center", va="bottom", fontsize=8)
    # barra de CD
    ax.hlines(-1.1, lo, lo + cd, color="#c92a2a", lw=2)
    ax.text(lo + cd / 2, -1.25, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8, color="#c92a2a")
    mitad = (k + 1) // 2
    for i, (nombre, v) in enumerate(r.items()):
        izquierda = i < mitad
        y = 0.6 + (i if izquierda else k - 1 - i) * 0.45
        x_txt = lo - 0.4 if izquierda else hi + 0.4
        ax.plot([v, v, x_txt], [0, y, y], color="#495057", lw=0.8)
        etiqueta = nombre if len(nombre) <= max_etiqueta else nombre[:max_etiqueta - 1] + "…"
        ax.text(x_txt + (-0.05 if izquierda else 0.05), y, f"{etiqueta} ({v:.2f})",
                ha="right" if izquierda else "left", va="center", fontsize=7.5)
    # grupos no significativos (cliques maximales de longitud < CD)
    vals = r.to_numpy()
    grupos = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in grupos):
            grupos.append((i, j))
    for g, (i, j) in enumerate(grupos):
        y = 0.25 + g * 0.12
        ax.hlines(y, vals[i] - 0.03, vals[j] + 0.03, color="#1c7ed6", lw=3)
    ax.axis("off")
    if titulo:
        ax.set_title(titulo, fontsize=10, fontweight="bold")
    return ax


# ----------------------------------------------------------------------------------------------
# Tamaños de efecto
# ----------------------------------------------------------------------------------------------
def delta_cliff(x, y) -> tuple[float, str]:
    """Delta de Cliff: P(X > Y) - P(X < Y), con la escala de Romano et al. (2006)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mayor = (x[:, None] > y[None, :]).mean()
    menor = (x[:, None] < y[None, :]).mean()
    d = float(mayor - menor)
    a = abs(d)
    mag = "despreciable" if a < 0.147 else "pequeño" if a < 0.33 else "mediano" if a < 0.474 else "grande"
    return d, mag


def d_cohen_pareado(dif) -> tuple[float, str]:
    """d de Cohen para diferencias pareadas (d_z = media / desviación estándar)."""
    dif = np.asarray(dif, float)
    d = float(dif.mean() / dif.std(ddof=1))
    a = abs(d)
    mag = "despreciable" if a < 0.2 else "pequeño" if a < 0.5 else "mediano" if a < 0.8 else "grande"
    return d, mag


# ----------------------------------------------------------------------------------------------
# Bootstrap por bloques de semana con intervalo BCa
# ----------------------------------------------------------------------------------------------
class AUCPonderado:
    """AUC (y AUC-PR) con pesos por observación en O(n): sirve para remuestrear sin reordenar."""

    def __init__(self, y, s):
        self.unicos, self.inv = np.unique(np.asarray(s, float), return_inverse=True)
        self.inv = self.inv.ravel()
        self.y = np.asarray(y, float)

    def auc(self, w):
        k = len(self.unicos)
        wp = np.bincount(self.inv, weights=w * self.y, minlength=k)
        wn = np.bincount(self.inv, weights=w * (1 - self.y), minlength=k)
        return float((wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


def bca_bloques(estadistico, bloques, B: int = 2000, semilla: int = 42, alfa: float = 0.05) -> dict:
    """Intervalo BCa con bootstrap por bloques (remuestrea bloques completos, p. ej. semanas).

    Parameters
    ----------
    estadistico : función que recibe un vector de pesos por observación y devuelve un número
        (los pesos son el número de veces que la observación sale en la réplica).
    bloques : etiqueta de bloque de cada observación.

    Notes
    -----
    El sesgo z0 sale de la proporción de réplicas por debajo del valor observado y la
    aceleración `a` del jackknife que deja fuera un bloque cada vez (Efron, 1987).
    """
    codigos, bl = np.unique(np.asarray(bloques), return_inverse=True)
    G = len(codigos)
    rng = np.random.default_rng(semilla)
    theta = estadistico(np.ones(len(bl)))
    reps = np.empty(B)
    for b in range(B):
        conteo = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        reps[b] = estadistico(conteo[bl])
    jack = np.empty(G)
    for g in range(G):
        w = np.ones(len(bl))
        w[bl == g] = 0.0
        jack[g] = estadistico(w)
    z0 = stats.norm.ppf(np.clip((reps < theta).mean() + 0.5 * (reps == theta).mean(), 1e-6, 1 - 1e-6))
    dj = jack.mean() - jack
    a = (dj ** 3).sum() / (6 * ((dj ** 2).sum()) ** 1.5 + 1e-300)
    za = stats.norm.ppf([alfa / 2, 1 - alfa / 2])
    adj = stats.norm.cdf(z0 + (z0 + za) / (1 - a * (z0 + za)))
    lo, hi = np.quantile(reps, adj)
    plo, phi = np.quantile(reps, [alfa / 2, 1 - alfa / 2])
    # p bilateral para H0: estadístico = 0 (útil cuando el estadístico es una diferencia): el doble
    # de la proporción de réplicas al otro lado del cero, con piso 1/B
    p0 = max(min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean())), 1 / B)
    return {"estimado": theta, "ic_inf": lo, "ic_sup": hi, "ic_inf_percentil": plo,
            "ic_sup_percentil": phi, "z0": z0, "aceleracion": a, "se": reps.std(ddof=1), "B": B,
            "p_boot": p0}


# ----------------------------------------------------------------------------------------------
# Regresión: Diebold-Mariano (HLN), Giacomini-White, Clark-West, bootstrap estacionario
# ----------------------------------------------------------------------------------------------
def _varianza_hac(x, rezagos: int | None = None) -> float:
    """Varianza de largo plazo de Newey-West (núcleo de Bartlett)."""
    x = np.asarray(x, float) - np.mean(x)
    T = len(x)
    if rezagos is None:
        rezagos = int(np.floor(4 * (T / 100) ** (2 / 9)))
    v = x @ x / T
    for l in range(1, rezagos + 1):
        v += 2 * (1 - l / (rezagos + 1)) * (x[l:] @ x[:-l]) / T
    return float(v)


def diebold_mariano(perdida1, perdida2, h: int = 1, rezagos: int | None = None) -> dict:
    """Diebold-Mariano con la corrección de Harvey, Leybourne y Newbold (1997).

    d_t = L1_t - L2_t (positivo: el modelo 2 pierde menos). La varianza usa Newey-West para
    tolerar autocorrelación de la serie de diferencias; el estadístico corregido se compara con
    una t de Student con T-1 grados de libertad.
    """
    d = np.asarray(perdida1, float) - np.asarray(perdida2, float)
    T = len(d)
    var = _varianza_hac(d, rezagos)
    dm = d.mean() / np.sqrt(var / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    est = dm * hln
    return {"d_media": d.mean(), "DM": dm, "DM_HLN": est, "p": 2 * stats.t.sf(abs(est), T - 1), "T": T}


def giacomini_white(perdida1, perdida2, rezagos_instrumento: int = 1) -> dict:
    """Prueba de habilidad predictiva condicional de Giacomini y White (2006), horizonte 1.

    Instrumentos h_{t-1} = (1, d_{t-1}): se contrasta E[h_{t-1} d_t] = 0, es decir, que la
    diferencia de pérdidas no sea predecible con su propio pasado (además de tener media cero).
    Estadístico de Wald T Zbar' Omega^-1 Zbar ~ chi2(q).
    """
    d = np.asarray(perdida1, float) - np.asarray(perdida2, float)
    q = 1 + rezagos_instrumento
    filas = [np.r_[1.0, d[t - rezagos_instrumento:t][::-1]] for t in range(rezagos_instrumento, len(d))]
    H = np.asarray(filas)
    Z = H * d[rezagos_instrumento:, None]
    T = len(Z)
    zbar = Z.mean(axis=0)
    omega = (Z - zbar).T @ (Z - zbar) / T
    est = float(T * zbar @ np.linalg.pinv(omega) @ zbar)
    return {"GW": est, "gl": q, "p": stats.chi2.sf(est, q), "T": T,
            "signo": "modelo 2 mejor" if d.mean() > 0 else "modelo 1 mejor"}


def clark_west(y, pred_restringido, pred_general, bloques=None) -> dict:
    """Prueba de Clark y West (2007) para modelos anidados (H1: el modelo general es mejor).

    f_t = e1_t^2 - [e2_t^2 - (yhat1_t - yhat2_t)^2]; el término de ajuste elimina el sesgo del
    DM a favor del modelo restringido bajo H0. Si se dan `bloques` (semanas), f se promedia por
    bloque y se usa la varianza de Newey-West de esa serie. Prueba unilateral.
    """
    y, p1, p2 = (np.asarray(v, float) for v in (y, pred_restringido, pred_general))
    f = (y - p1) ** 2 - ((y - p2) ** 2 - (p1 - p2) ** 2)
    if bloques is not None:
        f = pd.Series(f).groupby(np.asarray(bloques)).mean().to_numpy()
    T = len(f)
    t = f.mean() / np.sqrt(_varianza_hac(f) / T)
    return {"f_media": f.mean(), "CW": t, "p": stats.norm.sf(t), "T": T}


def ic_bootstrap_estacionario(x, estadistico=np.mean, B: int = 2000, semilla: int = 42,
                              tam_bloque: float | None = None) -> dict:
    """Intervalo de confianza con el bootstrap estacionario de Politis y Romano (1994)."""
    from arch.bootstrap import StationaryBootstrap, optimal_block_length
    x = np.asarray(x, float)
    if tam_bloque is None:
        tam_bloque = float(max(1.0, optimal_block_length(x)["stationary"].iloc[0]))
    bs = StationaryBootstrap(tam_bloque, x, seed=semilla)
    ic = bs.conf_int(estadistico, reps=B, method="percentile")
    return {"estimado": float(estadistico(x)), "ic_inf": float(ic[0, 0]), "ic_sup": float(ic[1, 0]),
            "bloque": tam_bloque}


def model_confidence_set(perdidas: pd.DataFrame, alfa: float = 0.10, B: int = 2000,
                         semilla: int = 42, tam_bloque: int | None = None) -> pd.DataFrame:
    """Model Confidence Set de Hansen, Lunde y Nason (2011) con bootstrap estacionario.

    `perdidas`: T × M (filas = periodos, columnas = modelos). Devuelve por modelo el p-valor MCS,
    si pertenece al conjunto de confianza al nivel alfa y su pérdida promedio.
    """
    from arch.bootstrap import MCS, optimal_block_length
    if tam_bloque is None:
        tam_bloque = int(np.ceil(optimal_block_length(perdidas.mean(axis=1))["stationary"].iloc[0]))
    mcs = MCS(perdidas, size=alfa, reps=B, block_size=max(tam_bloque, 1), method="R",
              bootstrap="stationary", seed=semilla)
    mcs.compute()
    pv = mcs.pvalues.iloc[:, 0]
    out = pd.DataFrame({"perdida_media": perdidas.mean(), "rango_promedio": perdidas.rank(axis=1).mean(),
                        "p_mcs": pv.reindex(perdidas.columns)})
    out["en_conjunto"] = out.index.isin(mcs.included)
    return out.sort_values("perdida_media")


def spa_hansen(perdida_benchmark, perdidas_modelos: pd.DataFrame, B: int = 2000, semilla: int = 42,
               tam_bloque: int | None = None) -> dict:
    """Superior Predictive Ability de Hansen (2005) y Reality Check de White (2000).

    H0: ningún modelo supera al benchmark. Devuelve los p-valores inferior, consistente (SPA) y
    superior (equivalente al Reality Check).
    """
    from arch.bootstrap import SPA, optimal_block_length
    if tam_bloque is None:
        tam_bloque = int(np.ceil(optimal_block_length(np.asarray(perdida_benchmark))["stationary"].iloc[0]))
    spa = SPA(np.asarray(perdida_benchmark), perdidas_modelos, block_size=max(tam_bloque, 1), reps=B,
              bootstrap="stationary", seed=semilla)
    spa.compute()
    pv = spa.pvalues
    return {"p_inferior": float(pv["lower"]), "p_spa": float(pv["consistent"]),
            "p_reality_check": float(pv["upper"]), "bloque": tam_bloque}


# ----------------------------------------------------------------------------------------------
# Pruebas sobre residuos de regresión
# ----------------------------------------------------------------------------------------------
def prueba_white(residuos, exog) -> dict:
    """Prueba de White: regresión de e^2 sobre las variables, sus cuadrados y productos cruzados."""
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_white
    X = sm.add_constant(np.asarray(exog, float), has_constant="add")
    lm, p_lm, f, p_f = het_white(np.asarray(residuos, float), X)
    return {"LM": lm, "p": p_lm, "F": f, "p_F": p_f}


def prueba_bds(serie, max_dim: int = 5) -> pd.DataFrame:
    """Prueba BDS de independencia (Brock, Dechert, Scheinkman y LeBaron) para dimensiones 2..m."""
    from statsmodels.tsa.stattools import bds
    est, p = bds(np.asarray(serie, float), max_dim=max_dim)
    return pd.DataFrame({"dimension": range(2, max_dim + 1), "BDS": est, "p": p})


def normalidad(residuos) -> dict:
    r = np.asarray(residuos, float)
    jb = stats.jarque_bera(r)
    k2 = stats.normaltest(r)
    return {"asimetria": float(stats.skew(r)), "curtosis_exceso": float(stats.kurtosis(r)),
            "JB": float(jb.statistic), "p_JB": float(jb.pvalue), "K2": float(k2.statistic),
            "p_K2": float(k2.pvalue)}
