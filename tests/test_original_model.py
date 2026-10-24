"""Pruebas unitarias del modelo original AJ-KL (apilamiento jerárquico con encogimiento KL y olvido)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from original_model import (ApilamientoJerarquicoKL, brecha_fw, descenso_espejo, kl,  # noqa: E402
                            logistica_newton_ponderada, objetivo_nodo)


def datos_regiones(n=6000, semilla=0):
    """Dos regiones: en la A el modelo 0 es informativo y el 1 es ruido; en la B, al revés."""
    rng = np.random.default_rng(semilla)
    region = rng.integers(0, 2, n)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-2.5 * z))).astype(int)
    bueno = 1 / (1 + np.exp(-(2.5 * z + rng.normal(0, 0.5, n))))
    malo = 1 / (1 + np.exp(-rng.normal(0, 1, n)))
    P = np.where(region[:, None] == 0, np.c_[bueno, malo], np.c_[malo, bueno])
    return np.c_[P, region], y, region


def test_recupera_los_pesos_de_cada_region():
    X, y, region = datos_regiones()
    m = ApilamientoJerarquicoKL(lam=1.0, profundidad=2, n_modelos=2, calibrar=False).fit(X, y)
    assert m.pesos_[0][0][0] > 0.8          # región A: domina el modelo 0
    assert m.pesos_[0][1][1] > 0.8          # región B: domina el modelo 1
    glob = ApilamientoJerarquicoKL(lam=1.0, profundidad=1, n_modelos=2, calibrar=False).fit(X, y)
    assert roc_auc_score(y, m.predict_proba(X)[:, 1]) > roc_auc_score(y, glob.predict_proba(X)[:, 1]) + 0.03


def test_descenso_espejo_alcanza_el_optimo_de_referencia():
    rng = np.random.default_rng(1)
    P = np.clip(rng.random((800, 4)), 1e-4, 1 - 1e-4)
    y = (rng.random(800) < P[:, 2]).astype(float)
    a = rng.uniform(0.5, 1.0, 800)
    u = np.array([0.4, 0.3, 0.2, 0.1])
    mu = 0.05
    w, hist = descenso_espejo(P, y, a, u, mu, eta0=1.0, max_iter=3000, tol=1e-12)
    # referencia: SLSQP sobre el símplex
    ref = minimize(lambda v: objetivo_nodo(np.clip(v, 1e-12, None) / np.clip(v, 1e-12, None).sum(), P, y, a, u, mu),
                   u, method="SLSQP", bounds=[(1e-9, 1)] * 4, constraints=({"type": "eq", "fun": lambda v: v.sum() - 1},),
                   options={"ftol": 1e-12, "maxiter": 500})
    assert objetivo_nodo(w, P, y, a, u, mu) <= ref.fun + 1e-4
    assert abs(w.sum() - 1) < 1e-12 and np.all(w > 0)
    assert hist.min() == pytest.approx(objetivo_nodo(w, P, y, a, u, mu))


def test_converge_con_modelos_correlacionados_y_valores_por_defecto():
    """Régimen real: 9 modelos casi iguales (objetivo plano) y encogimiento débil; se exige el certificado."""
    rng = np.random.default_rng(3)
    n = 20000
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-1.2 * z))).astype(float)
    P = np.clip(1 / (1 + np.exp(-(1.2 * z[:, None] + rng.normal(0, 0.3, (n, 9)) * np.linspace(0.5, 2, 9)))), 1e-4, 1 - 1e-4)
    a = np.ones(n)
    u = np.full(9, 1 / 9)
    mu = 1e-4
    w, _ = descenso_espejo(P, y, a, u, mu)
    ref = minimize(lambda v: objetivo_nodo(np.exp(v - v.max()) / np.exp(v - v.max()).sum(), P, y, a, u, mu),
                   np.zeros(9), method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5000})
    w_ref = np.exp(ref.x - ref.x.max()) / np.exp(ref.x - ref.x.max()).sum()
    assert objetivo_nodo(w, P, y, a, u, mu) <= objetivo_nodo(w_ref, P, y, a, u, mu) + 1e-7
    assert brecha_fw(w, P, y, a, u, mu) < 1e-5
    assert np.abs(w - w_ref).sum() < 0.05


def test_casos_limite_de_lambda():
    X, y, _ = datos_regiones(3000)
    fuerte = ApilamientoJerarquicoKL(lam=1e7, profundidad=2, n_modelos=2, calibrar=False).fit(X, y)
    np.testing.assert_allclose(fuerte.raiz_, [0.5, 0.5], atol=1e-2)            # λ → ∞: promedio simple
    np.testing.assert_allclose(fuerte.pesos_[0][0], fuerte.raiz_, atol=1e-2)   # los hijos copian al padre
    libre = ApilamientoJerarquicoKL(lam=0.0, profundidad=2, n_modelos=2, calibrar=False).fit(X, y)
    assert libre.pesos_[0][0][0] > fuerte.pesos_[0][0][0]                       # λ = 0: apilamiento local


def test_kl_es_cero_solo_en_el_mismo_punto():
    u = np.array([0.2, 0.3, 0.5])
    assert kl(u, u) == pytest.approx(0.0)
    assert kl(np.array([0.3, 0.3, 0.4]), u) > 0


def test_el_olvido_favorece_el_regimen_reciente():
    rng = np.random.default_rng(2)
    n = 4000
    t = np.repeat(np.arange(40), n // 40)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-2 * z))).astype(int)
    bueno = 1 / (1 + np.exp(-2 * z))
    malo = 1 / (1 + np.exp(-rng.normal(size=n)))
    reciente = t >= 30                           # el modelo 1 es el bueno solo en las últimas semanas
    P = np.where(reciente[:, None], np.c_[malo, bueno], np.c_[bueno, malo])
    sin = ApilamientoJerarquicoKL(lam=0.1, profundidad=1, gamma=1.0, calibrar=False).fit(P, y, tiempos=t)
    con = ApilamientoJerarquicoKL(lam=0.1, profundidad=1, gamma=0.8, calibrar=False).fit(P, y, tiempos=t)
    assert con.raiz_[1] > sin.raiz_[1] + 0.2


def test_nodo_no_visto_usa_el_padre():
    X, y, _ = datos_regiones(2000)
    m = ApilamientoJerarquicoKL(lam=1.0, profundidad=2, n_modelos=2, calibrar=False).fit(X, y)
    X_nuevo = X[:5].copy()
    X_nuevo[:, 2] = 99
    np.testing.assert_allclose(m.pesos_por_fila(X_nuevo), np.tile(m.raiz_, (5, 1)))


def test_calibracion_beta_mejora_un_pool_descalibrado():
    rng = np.random.default_rng(3)
    z = rng.normal(size=5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-3 * z))).astype(int)
    p1 = 1 / (1 + np.exp(-0.5 * z))               # demasiado conservador
    p2 = 1 / (1 + np.exp(-0.6 * z))
    P = np.c_[p1, p2]
    cruda = ApilamientoJerarquicoKL(profundidad=1, calibrar=False).fit(P[:3000], y[:3000])
    cal = ApilamientoJerarquicoKL(profundidad=1, calibrar=True).fit(P[:3000], y[:3000])
    assert log_loss(y[3000:], cal.predict_proba(P[3000:])[:, 1]) < log_loss(y[3000:], cruda.predict_proba(P[3000:])[:, 1]) - 0.05


def test_newton_ponderado_coincide_con_sklearn():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(4)
    F = rng.normal(size=(3000, 2))
    y = rng.random(3000) < 1 / (1 + np.exp(-(F @ [1.0, -0.7] + 0.2)))
    peso = rng.uniform(0.2, 1.0, 3000)
    w = logistica_newton_ponderada(F, y, peso, ridge=0.0)
    ref = LogisticRegression(penalty=None, max_iter=5000).fit(F, y, sample_weight=peso / peso.mean())
    np.testing.assert_allclose(w, np.r_[ref.intercept_, ref.coef_[0]], atol=1e-4)


def test_reproducible():
    X, y, _ = datos_regiones(1500)
    t = np.arange(len(y)) // 50
    p1 = ApilamientoJerarquicoKL(gamma=0.97, profundidad=2, n_modelos=2).fit(X, y, tiempos=t).predict_proba(X)
    p2 = ApilamientoJerarquicoKL(gamma=0.97, profundidad=2, n_modelos=2).fit(X, y, tiempos=t).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_compatible_con_clone():
    m = ApilamientoJerarquicoKL(lam=3.0, gamma=0.98, profundidad=2)
    c = clone(m)
    assert c.get_params()["lam"] == 3.0 and c.get_params()["gamma"] == 0.98
    c.set_params(profundidad=3)
    assert c.profundidad == 3


@pytest.mark.parametrize("cambio", [dict(lam=-1), dict(gamma=0), dict(gamma=1.2), dict(profundidad=5),
                                    dict(eta0=0), dict(max_iter=0), dict(epsilon=0.6)])
def test_hiperparametros_invalidos(cambio):
    X, y, _ = datos_regiones(300)
    with pytest.raises(ValueError):
        ApilamientoJerarquicoKL(n_modelos=2, **cambio).fit(X, y)


def test_entradas_invalidas():
    X, y, _ = datos_regiones(300)
    m = ApilamientoJerarquicoKL(n_modelos=2)
    with pytest.raises(ValueError):
        m.fit(X, y[:-1])
    with pytest.raises(ValueError):
        m.fit(X, y * 3)
    malo = X.copy()
    malo[0, 0] = 1.5
    with pytest.raises(ValueError):
        m.fit(malo, y)
    with pytest.raises(ValueError):
        m.fit(X, y, tiempos=np.arange(10))
    m.fit(X, y)
    with pytest.raises(ValueError):
        m.predict_proba(X[:, :2])


def test_falla_si_no_esta_ajustado():
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        ApilamientoJerarquicoKL().predict_proba(np.full((2, 3), 0.5))
