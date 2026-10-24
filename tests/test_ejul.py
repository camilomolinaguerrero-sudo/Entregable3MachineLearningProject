"""Pruebas unitarias del modelo original (EJUL)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ejul import (CodificadorJerarquico, EnsambleUmbralesLatentes, logistica_newton,  # noqa: E402
                            nucleo, pesos_polinomio_local, platt_newton, proyeccion_isotonica,
                            rasgos_calibracion)


def datos_sinteticos(n=3000, semilla=0):
    """R = 1 + 0,8 x0 − 0,6 x1 + efecto de grupo + ruido; evento Y = 1{R < 0,5}."""
    rng = np.random.default_rng(semilla)
    X = rng.normal(size=(n, 4))
    g = rng.integers(0, 30, n)
    padre = g // 6
    efecto = rng.normal(0, 0.7, 30)
    R = 1 + 0.8 * X[:, 0] - 0.6 * X[:, 1] + efecto[g] + rng.normal(0, 0.6, n)
    y = (R < 0.5).astype(int)
    return np.c_[X, g, padre], y, R


def base_rapida():
    return LogisticRegression(max_iter=500)


# ------------------------------------------------------------------------- piezas algorítmicas
def test_isotonica_coincide_con_pav_de_referencia():
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(1)
    M = rng.random((50, 7))
    P = proyeccion_isotonica(M)
    ref = np.vstack([IsotonicRegression().fit_transform(np.arange(7), fila) for fila in M])
    np.testing.assert_allclose(P, ref, atol=1e-12)


def test_isotonica_deja_igual_un_vector_monotono_y_no_expande():
    v = np.array([[0.1, 0.2, 0.2, 0.5, 0.9]])
    np.testing.assert_allclose(proyeccion_isotonica(v), v)
    rng = np.random.default_rng(2)
    M = rng.random((200, 5))
    verdadero = np.sort(rng.random((200, 5)), axis=1)      # vectores del cono
    d_antes = np.linalg.norm(M - verdadero, axis=1)
    d_despues = np.linalg.norm(proyeccion_isotonica(M) - verdadero, axis=1)
    assert np.all(d_despues <= d_antes + 1e-12)             # propiedad P2


@pytest.mark.parametrize("tipo", ["uniforme", "epanechnikov", "triangular"])
@pytest.mark.parametrize("grado", [0, 1, 2])
def test_pesos_reproducen_polinomios(tipo, grado):
    u = np.linspace(-1, 1, 7)
    w = pesos_polinomio_local(u, nucleo(u, tipo), grado)
    assert w.sum() == pytest.approx(1.0)
    for j in range(1, grado + 1):
        assert w @ u ** j == pytest.approx(0.0, abs=1e-12)
    # diseño simétrico: el momento de orden 1 se anula siempre
    assert w @ u == pytest.approx(0.0, abs=1e-12)


def test_platt_recupera_parametros():
    rng = np.random.default_rng(3)
    z = rng.normal(size=20000)
    y = rng.random(20000) < 1 / (1 + np.exp(-(1.7 * z - 0.4)))
    a, b = platt_newton(z, y)
    assert a == pytest.approx(1.7, abs=0.08)
    assert b == pytest.approx(-0.4, abs=0.06)


def test_logistica_newton_coincide_con_sklearn():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(4)
    F = rng.normal(size=(5000, 3))
    y = rng.random(5000) < 1 / (1 + np.exp(-(F @ [1.0, -0.5, 0.2] + 0.3)))
    w = logistica_newton(F, y, ridge=0.0)
    ref = LogisticRegression(penalty=None, max_iter=2000).fit(F, y)
    np.testing.assert_allclose(w, np.r_[ref.intercept_, ref.coef_[0]], atol=1e-4)


def test_calibracion_beta_contiene_a_la_identidad():
    # con a = b = 1 y c = 0, la calibración beta es la identidad
    p = np.linspace(0.01, 0.99, 50)
    R = rasgos_calibracion(p, "beta")
    from scipy.special import expit
    np.testing.assert_allclose(expit(R @ [1.0, 1.0]), p, atol=1e-12)


def test_codificador_contrae_hacia_el_padre():
    # grupo 0: 2 de 2 eventos (pocos datos); grupo 1: 50 de 100
    G = np.array([[0, 0]] * 2 + [[1, 0]] * 100)
    y = np.r_[1, 1, np.ones(50), np.zeros(50)]
    cod = CodificadorJerarquico().fit(G, y)
    t = cod.tasas(np.array([[0, 0], [1, 0], [99, 0], [-1, -1]]))
    assert 0.5 < t[0, 0] < 1.0              # contraído hacia la tasa del padre, no 1
    assert t[1, 0] == pytest.approx(0.5, abs=0.02)
    assert t[2, 0] == pytest.approx(t[2, 1])  # grupo no visto: hereda la tasa del padre
    assert t[3, 1] == pytest.approx(cod.p0_)  # sin grupo: tasa global


# ------------------------------------------------------------------------- estimador
def test_aprende_y_supera_al_clasificador_sin_latente():
    X, y, R = datos_sinteticos()
    tr, te = slice(0, 2000), slice(2000, None)
    kw = dict(estimador_base=base_rapida(), columnas_grupo=(4, 5), random_state=0)
    ejul = EnsambleUmbralesLatentes(h=0.3, **kw).fit(X[tr], y[tr], latente=R[tr])
    simple = EnsambleUmbralesLatentes(h=0.0, jerarquico=False, calibrar=False, **kw).fit(X[tr], y[tr])
    auc_ejul = roc_auc_score(y[te], ejul.predict_proba(X[te])[:, 1])
    auc_simple = roc_auc_score(y[te], simple.predict_proba(X[te])[:, 1])
    assert auc_ejul > 0.85
    assert auc_ejul > auc_simple


class LogisticaEncogida(LogisticRegression):
    """Clasificador deliberadamente descalibrado (probabilidades encogidas hacia 0,5)."""

    def predict_proba(self, X):
        from scipy.special import expit, logit
        p = expit(0.35 * logit(super().predict_proba(X)[:, 1]))
        return np.c_[1 - p, p]


@pytest.mark.parametrize("calibrador", ["platt", "beta"])
def test_calibradores_corrigen_un_modelo_descalibrado(calibrador):
    from sklearn.metrics import log_loss
    X, y, R = datos_sinteticos(3000, semilla=5)
    kw = dict(estimador_base=LogisticaEncogida(max_iter=500), columnas_grupo=(4, 5), h=0.3, random_state=0)
    cal = EnsambleUmbralesLatentes(calibrador=calibrador, **kw).fit(X[:2000], y[:2000], latente=R[:2000])
    ll_cal = log_loss(y[2000:], cal.predict_proba(X[2000:])[:, 1])
    ll_crudo = log_loss(y[2000:], cal.probabilidad_cruda(X[2000:]))
    assert ll_cal < ll_crudo - 0.02


def test_reproducible_con_la_misma_semilla():
    X, y, R = datos_sinteticos(800)
    kw = dict(estimador_base=base_rapida(), columnas_grupo=(4, 5), random_state=7)
    p1 = EnsambleUmbralesLatentes(**kw).fit(X, y, latente=R).predict_proba(X)
    p2 = EnsambleUmbralesLatentes(**kw).fit(X, y, latente=R).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_probabilidades_validas_y_umbrales_monotonos():
    X, y, R = datos_sinteticos(800)
    m = EnsambleUmbralesLatentes(estimador_base=base_rapida(), columnas_grupo=(4, 5), n_umbrales=7).fit(X, y, latente=R)
    P = m.predict_proba(X)
    assert P.shape == (len(X), 2)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert np.all((P >= 0) & (P <= 1))
    F = m.probabilidades_umbral(X)
    assert F.shape == (len(X), 7)
    assert np.all(np.diff(F, axis=1) >= -1e-12)
    assert set(m.predict(X)) <= {0, 1}


def test_sin_latente_se_reduce_a_un_miembro():
    X, y, _ = datos_sinteticos(600)
    m = EnsambleUmbralesLatentes(estimador_base=base_rapida(), columnas_grupo=(4, 5)).fit(X, y)
    assert len(m.miembros_) == 1 and m.pesos_.tolist() == [1.0]


def test_pliegues_temporales_externos():
    X, y, R = datos_sinteticos(900)
    plie = [(np.arange(0, 300), np.arange(300, 600)), (np.arange(0, 600), np.arange(600, 900))]
    m = EnsambleUmbralesLatentes(estimador_base=base_rapida(), columnas_grupo=(4, 5)).fit(X, y, latente=R, pliegues=plie)
    assert np.isnan(m.oof_[:300]).all() and np.isfinite(m.oof_[300:]).all()


def test_compatible_con_clone_y_get_params():
    m = EnsambleUmbralesLatentes(h=0.15, n_umbrales=3, nucleo="triangular")
    c = clone(m)
    assert c.get_params()["h"] == 0.15 and c.get_params()["nucleo"] == "triangular"
    c.set_params(grado=2)
    assert c.grado == 2


@pytest.mark.parametrize("cambio", [dict(n_umbrales=4), dict(n_umbrales=0), dict(h=-0.1),
                                    dict(nucleo="gauss"), dict(grado=3), dict(fuerza_eb=0),
                                    dict(pliegues_internos=1), dict(calibrador="isotonica"), dict(prediccion="x"),
                                    dict(prediccion="pliegues", calibrar=False)])
def test_hiperparametros_invalidos(cambio):
    X, y, R = datos_sinteticos(300)
    with pytest.raises(ValueError):
        EnsambleUmbralesLatentes(estimador_base=base_rapida(), **cambio).fit(X, y, latente=R)


def test_entradas_invalidas():
    X, y, R = datos_sinteticos(300)
    m = EnsambleUmbralesLatentes(estimador_base=base_rapida(), columnas_grupo=(4, 5))
    with pytest.raises(ValueError):
        m.fit(X, np.r_[y, 0])                       # largo distinto
    with pytest.raises(ValueError):
        m.fit(X, y * 2)                              # no binaria
    with pytest.raises(ValueError):
        m.fit(X, y, latente=R[:-1])                  # latente de otro largo
    with pytest.raises(ValueError):
        m.fit(X, y, latente=np.r_[R[:-1], np.inf])   # latente no finita
    m.fit(X, y, latente=R)
    with pytest.raises(ValueError):
        m.predict_proba(X[:, :5])                    # columnas faltantes


def test_falla_si_no_esta_ajustado():
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        EnsambleUmbralesLatentes().predict_proba(np.zeros((2, 3)))


def test_indices_negativos_equivalen_a_positivos():
    X, y, R = datos_sinteticos(600)
    kw = dict(estimador_base=base_rapida(), random_state=3)
    p_pos = EnsambleUmbralesLatentes(columnas_grupo=(4, 5), **kw).fit(X, y, latente=R).predict_proba(X)
    p_neg = EnsambleUmbralesLatentes(columnas_grupo=(-2, -1), **kw).fit(X, y, latente=R).predict_proba(X)
    np.testing.assert_allclose(p_pos, p_neg)


def test_prediccion_por_pliegues():
    X, y, R = datos_sinteticos(900)
    m = EnsambleUmbralesLatentes(estimador_base=base_rapida(), columnas_grupo=(4, 5), prediccion="pliegues").fit(X, y, latente=R)
    assert len(m.miembros_pliegue_) == 3
    p = m.predict_proba(X)[:, 1]
    assert np.all((p > 0) & (p < 1)) and roc_auc_score(y, p) > 0.8


def test_codificacion_temporal_solo_usa_el_pasado():
    # grupo 0 con eventos solo en el periodo 2: en los periodos 0 y 1 su tasa debe ser la del prior
    G = np.array([[0, 0]] * 6)
    y = np.array([0, 0, 0, 0, 1, 1])
    t = np.array([0, 0, 1, 1, 2, 2])
    cod = CodificadorJerarquico().fit(G, y)
    T = cod.tasas_ordenadas(G, y, t)
    assert np.allclose(T[0, 0], T[1, 0])                 # mismo periodo, misma información
    assert T[4, 0] < cod.tasas(G)[4, 0]                   # en t = 2 aún no se conocen sus propios eventos
    # cambiar las etiquetas del último periodo no altera la codificación de los anteriores
    y2 = y.copy(); y2[4:] = 0
    cod2 = CodificadorJerarquico().fit(G, y2)
    cod2.niveles_ = cod.niveles_; cod2.p0_ = cod.p0_
    assert np.allclose(cod2.tasas_ordenadas(G, y2, t)[:4], T[:4])


def test_codificacion_temporal_requiere_tiempos():
    X, y, R = datos_sinteticos(300)
    with pytest.raises(ValueError):
        EnsambleUmbralesLatentes(estimador_base=base_rapida(), columnas_grupo=(4, 5), codificacion="temporal").fit(X, y, latente=R)
    tiempos = np.repeat(np.arange(30), 10)
    m = EnsambleUmbralesLatentes(estimador_base=base_rapida(), columnas_grupo=(4, 5), codificacion="temporal").fit(
        X, y, latente=R, tiempos=tiempos)
    assert m.predict_proba(X).shape == (300, 2)
