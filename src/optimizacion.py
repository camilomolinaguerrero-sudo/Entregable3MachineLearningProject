"""Etapa 3: los cuatro métodos de optimización de hiperparámetros sobre una función objetivo común.

La función objetivo (`Objetivo`) evalúa una configuración con validación cruzada interna. El
preprocesamiento de cada pliegue interno se ajusta SOLO con su parte de entrenamiento y el
remuestreo (SMOTE/ADASYN) se aplica SOLO a esa parte; como ninguno de los dos depende de los
hiperparámetros del modelo, se calcula una vez por pliegue y se reutiliza en todas las
evaluaciones (es lo mismo que hace `Pipeline(memory=...)` de scikit-learn: el resultado es
idéntico al de reajustar el pipeline completo en cada evaluación, pero sin repetir trabajo).

Los optimizadores reciben el mismo espacio (`modelos.Param`) y el mismo presupuesto de
evaluaciones y devuelven el historial completo (configuración, puntaje, tiempo) para construir las
curvas de desempeño en cualquier momento (anytime performance).
"""
from __future__ import annotations

import itertools
import random
import time

import numpy as np
import optuna
from deap import base, creator, tools
from sklearn.metrics import mean_squared_error, roc_auc_score

import modelos as M

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Parámetros de la Optimización Bayesiana (TPE)
TPE_ARRANQUE = 8          # evaluaciones aleatorias iniciales (un tercio del presupuesto)
TPE_CANDIDATOS = 24       # candidatos muestreados de l(x) para maximizar la Mejora Esperada

# Parámetros del Algoritmo Genético
AG_MU = 6                 # tamaño de la población (padres)
AG_LAMBDA = 6             # hijos por generación
AG_TORNEO = 3             # tamaño del torneo de selección
AG_PCX = 0.7              # probabilidad de cruce (SBX acotado, eta = 15)
AG_PMUT = 0.3             # probabilidad de mutación (polinomial acotada, eta = 20, indpb = 1/d)
AG_ETA_CX = 15.0
AG_ETA_MUT = 20.0


def _clave(params: dict) -> tuple:
    """Clave de caché con 12 cifras significativas (relativas, para no mezclar valores pequeños
    como var_smoothing = 1e-12 y 1e-11)."""
    return tuple(sorted((k, float(f"{v:.12g}") if isinstance(v, float) else v) for k, v in params.items()))


class Objetivo:
    """Evalúa configuraciones de un modelo con validación cruzada interna sobre pliegues en caché.

    Parameters
    ----------
    espec : especificación del modelo (`modelos.EspecModelo`).
    pliegues : lista de tuplas (X_tr, y_tr, X_va, y_va) ya preprocesadas y, si aplica,
        remuestreadas (solo la parte de entrenamiento).
    balanceo : técnica de balanceo (para `class_weight`).
    semilla : semilla de los estimadores.

    Notes
    -----
    El puntaje es el AUC-ROC medio de los pliegues (clasificación) o el RMSE medio con signo
    negativo (regresión), de modo que siempre se maximiza. Las configuraciones repetidas se
    responden desde la caché sin volver a entrenar.
    """

    def __init__(self, espec, pliegues, balanceo="Sin balanceo", semilla=M.SEMILLA):
        self.espec = espec
        self.pliegues = pliegues
        self.balanceo = balanceo
        self.semilla = semilla
        self.cache: dict = {}
        self.predicciones: dict = {}

    def __call__(self, params: dict) -> dict:
        clave = _clave(params)
        if clave in self.cache:
            return {**self.cache[clave], "repetida": True, "segundos": 0.0}
        t0 = time.perf_counter()
        puntajes, preds = [], []
        for X_tr, y_tr, X_va, y_va in self.pliegues:
            m = M.construir(self.espec, params, self.balanceo, y_tr, self.semilla)
            m.fit(X_tr, y_tr)
            if self.espec.tarea == "clf":
                s = M.puntaje(m, X_va)
                puntajes.append(roc_auc_score(y_va, s))
            else:
                s = m.predict(X_va)
                puntajes.append(-np.sqrt(mean_squared_error(y_va, s)))
            preds.append(s.astype(np.float32))
        r = {"puntaje": float(np.mean(puntajes)), "puntajes_pliegue": [float(v) for v in puntajes],
             "segundos": time.perf_counter() - t0, "repetida": False}
        self.cache[clave] = {k: v for k, v in r.items() if k not in ("segundos", "repetida")}
        self.predicciones[clave] = preds
        return r


def _registrar(historial, params, r, t_inicio, **extra):
    historial.append({"evaluacion": len(historial) + 1, "params": dict(params), **r,
                      "t_acumulado": time.perf_counter() - t_inicio, **extra})


# ----------------------------------------------------------------------------------------------
# Grid Search
# ----------------------------------------------------------------------------------------------
def grid_search(obj: Objetivo, espacio, presupuesto: int, semilla: int) -> dict:
    """Recorre el producto cartesiano de los niveles de la rejilla.

    El orden de recorrido se baraja con la semilla: Grid Search no define un orden, y sin barajar la
    curva "anytime" dependería del orden arbitrario de las dimensiones.
    """
    puntos = [dict(zip([p.nombre for p in espacio], v))
              for v in itertools.product(*[p.rejilla for p in espacio])]
    random.Random(semilla).shuffle(puntos)
    historial, t0 = [], time.perf_counter()
    for params in puntos[:presupuesto]:
        _registrar(historial, params, obj(params), t0)
    return {"historial": historial}


# ----------------------------------------------------------------------------------------------
# Random Search
# ----------------------------------------------------------------------------------------------
def random_search(obj: Objetivo, espacio, presupuesto: int, semilla: int) -> dict:
    """Muestreo uniforme (o log-uniforme en las escalas logarítmicas) e independiente.

    Igual que en los otros tres métodos, el presupuesto cuenta solo configuraciones nuevas: si el
    muestreo repite una configuración ya evaluada, se descarta y se muestrea otra.
    """
    rng = np.random.default_rng(semilla)
    historial, t0, intentos = [], time.perf_counter(), 0
    while len(historial) < presupuesto and intentos < 50 * presupuesto:
        intentos += 1
        genes = rng.random(len(espacio))
        params = {p.nombre: p.decodificar(g) for p, g in zip(espacio, genes)}
        r = obj(params)
        if not r["repetida"]:
            _registrar(historial, params, r, t0)
    return {"historial": historial}


# ----------------------------------------------------------------------------------------------
# Optimización Bayesiana con TPE (Optuna)
# ----------------------------------------------------------------------------------------------
def _sugerir(trial, p):
    if p.tipo == "cat":
        return trial.suggest_categorical(p.nombre, list(p.opciones))
    if p.tipo == "int":
        return trial.suggest_int(p.nombre, int(p.bajo), int(p.alto), log=p.log)
    return trial.suggest_float(p.nombre, p.bajo, p.alto, log=p.log)


def bayes_tpe(obj: Objetivo, espacio, presupuesto: int, semilla: int, sampler=None) -> dict:
    """Optimización Bayesiana con Tree-structured Parzen Estimator.

    Modelo sustituto: dos estimadores de densidad de Parzen, l(x) para las configuraciones del
    mejor cuantil gamma de puntajes y g(x) para el resto. Función de adquisición: la Mejora Esperada,
    que bajo TPE es proporcional a [gamma + (1 - gamma) g(x)/l(x)]^-1; se maximiza eligiendo, entre
    `TPE_CANDIDATOS` puntos muestreados de l(x), el de mayor cociente l(x)/g(x). Las primeras
    `TPE_ARRANQUE` evaluaciones son aleatorias (exploración inicial); `multivariate=True` modela
    las interacciones entre hiperparámetros con un núcleo conjunto.
    """
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=semilla, n_startup_trials=TPE_ARRANQUE,
                                             n_ei_candidates=TPE_CANDIDATOS, multivariate=True)
    estudio = optuna.create_study(direction="maximize", sampler=sampler)
    historial, t0, intentos = [], time.perf_counter(), 0
    # ask/tell: si TPE propone una configuración ya evaluada, se le informa su puntaje (para que
    # actualice l(x) y g(x)) pero no se descuenta del presupuesto.
    while len(historial) < presupuesto and intentos < 50 * presupuesto:
        intentos += 1
        trial = estudio.ask()
        params = {p.nombre: _sugerir(trial, p) for p in espacio}
        r = obj(params)
        estudio.tell(trial, r["puntaje"])
        if not r["repetida"]:
            _registrar(historial, params, r, t0)
    return {"historial": historial}


# ----------------------------------------------------------------------------------------------
# Algoritmo genético (DEAP)
# ----------------------------------------------------------------------------------------------
if not hasattr(creator, "AptitudMax"):
    creator.create("AptitudMax", base.Fitness, weights=(1.0,))
    creator.create("Individuo", list, fitness=creator.AptitudMax)


def diversidad(poblacion) -> float:
    """Distancia euclidiana media entre pares de genomas, normalizada por sqrt(d) (0 = clones)."""
    G = np.asarray([list(ind) for ind in poblacion], float)
    if len(G) < 2:
        return 0.0
    dif = G[:, None, :] - G[None, :, :]
    D = np.sqrt((dif ** 2).sum(-1))
    return float(D[np.triu_indices(len(G), 1)].mean() / np.sqrt(G.shape[1]))


def genetico(obj: Objetivo, espacio, presupuesto: int, semilla: int,
             mu: int = AG_MU, lam: int = AG_LAMBDA) -> dict:
    """Algoritmo genético (mu + lambda) con codificación real en [0, 1]^d.

    * Selección de padres: torneo de tamaño 3 (presión selectiva moderada, robusta a la escala del
      puntaje, a diferencia de la ruleta).
    * Cruce: SBX acotado (eta = 15) con probabilidad 0,7; mutación polinomial acotada (eta = 20,
      probabilidad por gen 1/d) con probabilidad 0,3. Ambos operadores respetan los límites [0, 1].
    * Elitismo: los sobrevivientes son los mu mejores entre padres e hijos, así que el mejor
      individuo nunca se pierde.
    * Presupuesto: solo cuentan las configuraciones nuevas (un hijo idéntico a un individuo ya
      evaluado se responde desde la caché).
    * Diversidad: distancia media entre genomas y fracción de configuraciones distintas por
      generación, para detectar convergencia prematura.
    """
    d = len(espacio)
    tb = base.Toolbox()
    tb.register("mate", tools.cxSimulatedBinaryBounded, eta=AG_ETA_CX, low=0.0, up=1.0)
    tb.register("mutate", tools.mutPolynomialBounded, eta=AG_ETA_MUT, low=0.0, up=1.0, indpb=1.0 / d)
    tb.register("select", tools.selTournament, tournsize=AG_TORNEO)
    historial, generaciones, t0 = [], [], time.perf_counter()
    usados = 0

    def decodificar(ind):
        return {p.nombre: p.decodificar(g) for p, g in zip(espacio, ind)}

    def evaluar(ind, gen):
        nonlocal usados
        params = decodificar(ind)
        r = obj(params)
        if not r["repetida"]:
            usados += 1
            _registrar(historial, params, r, t0, generacion=gen)
        ind.fitness.values = (r["puntaje"],)

    # DEAP usa el módulo random global: un solo flujo (sembrado aquí) genera la población inicial y
    # alimenta a los operadores, para que la corrida sea reproducible y sus sorteos independientes
    random.seed(semilla)
    pob = [creator.Individuo([random.random() for _ in range(d)]) for _ in range(mu)]
    for ind in pob:
        if usados < presupuesto:
            evaluar(ind, 0)
    generaciones.append({"generacion": 0, "diversidad": diversidad(pob),
                         "unicos": len({_clave(decodificar(i)) for i in pob}) / len(pob),
                         "mejor": max(i.fitness.values[0] for i in pob),
                         "media": float(np.mean([i.fitness.values[0] for i in pob])),
                         "evaluaciones": usados})
    gen = 0
    while usados < presupuesto and gen < 10 * presupuesto:
        gen += 1
        antes = usados
        hijos = [tb.clone(i) for i in tb.select(pob, lam)]
        for h1, h2 in zip(hijos[::2], hijos[1::2]):
            if random.random() < AG_PCX:
                tb.mate(h1, h2)
                del h1.fitness.values, h2.fitness.values
        for h in hijos:
            if random.random() < AG_PMUT:
                tb.mutate(h)
                del h.fitness.values
        for h in hijos:
            if not h.fitness.valid:
                if usados < presupuesto:
                    evaluar(h, gen)
                else:
                    h.fitness.values = (-np.inf,)
        pob = tools.selBest(pob + hijos, mu)
        if usados == antes and usados < presupuesto:
            # generación sin configuraciones nuevas (población convergida): se reemplaza el peor
            # individuo por un inmigrante aleatorio para recuperar diversidad
            inm = creator.Individuo([random.random() for _ in range(d)])
            evaluar(inm, gen)
            pob = tools.selBest(pob[:-1] + [inm], mu) if inm.fitness.valid else pob
        decod = [_clave(decodificar(i)) for i in pob]
        generaciones.append({"generacion": gen, "diversidad": diversidad(pob),
                             "unicos": len(set(decod)) / len(decod),
                             "mejor": max(i.fitness.values[0] for i in pob),
                             "media": float(np.mean([i.fitness.values[0] for i in pob])),
                             "evaluaciones": usados})
    return {"historial": historial, "generaciones": generaciones}


OPTIMIZADORES = {
    "Grid Search": grid_search,
    "Random Search": random_search,
    "Bayesiana (TPE)": bayes_tpe,
    "Genética (DEAP)": genetico,
}
