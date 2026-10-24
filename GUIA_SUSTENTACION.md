# Guía para la sustentación (rúbrica ProyectosFinalesML202630)

Orden sugerido de la exposición (unos 20 minutos): presentación de diapositivas (problema → datos →
benchmarks → literatura → AJ-KL → teoría → diseño → resultados → límites) y cierre con una demostración en vivo
del dashboard: tabla de líderes, predicción de un corredor real con el AJ-KL y la pestaña «Modelo original».
Abran la URL de Render unos minutos antes: el plan gratuito tarda cerca de un minuto en despertar.

## Criterio 1 · Problema y formulación matemática (10 %)

**¿Por qué clasificación y no regresión?** La decisión operativa es binaria (alertar o no a la central). La
razón continua se usó en el modelo exploratorio EJUL (anexo); la etiqueta con umbral 0,5 es la definición
acordada desde el Entregable 1.

**¿Por qué el AUC?** Es independiente del umbral y del desbalance moderado (28 %). Como la salida es una
probabilidad, reportamos además Brier, log-loss y ECE, y el F1 con el umbral elegido en la validación interna.

## Criterio 2 · Datos, EDA y fuga (15 %)

**¿Cómo evitaron la fuga?** (1) Todas las variables usan semanas hasta t. (2) Imputación, escalado y balanceo
van dentro del pipeline y se ajustan solo con el entrenamiento de cada pliegue. (3) Hueco de 8 semanas entre
entrenamiento y validación, porque el objetivo usa un promedio móvil de 8 semanas. (4) 2025 no se usó para
elegir hiperparámetros. (5) El combinador del AJ-KL solo ve semanas anteriores al corte de entrenamiento de
cada pliegue (en la versión 2.1 se quitaron las 8 semanas del hueco que la versión 2.0 incluía).

**¿Qué mostró el EDA?** Desbalance moderado, estacionalidad, choques de 2020 (confinamiento) y 2021 (paro),
dependencia espacial y ninguna variable con correlación de Spearman mayor que 0,20 en valor absoluto: el techo
de información explica por qué todos los modelos quedan cerca.

## Criterio 3 · Fundamentos teóricos (25 %)

**Formulación.** q_i = w_rᵀ p_i con w_r en el símplex; J_r(w) = Σ γ^(t_max−t_i) ℓ(y_i, wᵀp_i) + λ KL(w‖w_padre).

**Función de pérdida.** Log-loss: regla de puntuación propia; premia probabilidades bien calibradas.

**Regularización.** KL hacia los pesos del padre (en la raíz, hacia la uniforme). Efectivamente μ_r = λ/S_r:
pesa más en nodos con poca muestra efectiva.

**Fundamento estadístico.** Es el MAP de una verosimilitud de Bernoulli ponderada con prior entrópico
exp(−λ KL(w‖w_padre)): Bayes empírico jerárquico sobre los pesos.

**Optimización.** Descenso espejo compuesto con la entropía: el paso tiene forma cerrada (proposición 2) y la
KL hacia el padre entra exacta. Paso adaptativo por retroceso, paso de Frank-Wolfe cuando el multiplicativo se
estanca, parada por la brecha de Frank-Wolfe (proposición 3a: F(w) − F* ≤ brecha).

**¿Por qué no L-BFGS o Newton?** Funcionarían (con M = 9 la hessiana es barata). Descenso espejo respeta el
símplex sin proyecciones ni reparametrización, y su paso es interpretable (actualización multiplicativa,
como los pesos exponenciales de la literatura de expertos). Usamos L-BFGS como referencia en las pruebas.

**Supuestos.** Probabilidades base fuera de muestra; jerarquía anidada (el código del corredor incluye la
central y el grupo, así que el anidamiento se cumple por construcción); la relación cambia lentamente en el
tiempo (olvido con semivida log 2 / log(1/γ)).

**Complejidad.** Entrenamiento O(L (n log n + K n M)), inferencia O(L log N + M) por fila. Medido: unos 10 s
para 375 mil filas y unos 2 ms por cada mil predicciones del combinador (sin contar los modelos base).

**Generalización.** Rademacher por nodo ≤ √(2 log M / n_r) (lema de Massart): la capacidad crece con log M,
no con el número de variables; la KL evita que los nodos pequeños se aparten del padre.

**Casos límite.** λ → ∞: promedio simple; λ → 0⁺: apilamiento local (el más cercano al padre en KL);
profundidad 1: apilamiento global; γ = 1: sin olvido.

**Pregunta difícil: «¿no es solo stacking?»** El stacking global es un caso particular (profundidad 1). Lo
nuevo es la jerarquía anidada propia del problema, el encogimiento KL hacia el padre, el olvido y el estimador
convexo con certificado. Y la ablación lo compara contra stacking logístico y BLP con el mismo historial.

## Criterio 4 · Implementación y diseño experimental (15 %)

**¿Es implementación propia?** Sí: `src/original_model.py` usa solo numpy y scipy, hereda de BaseEstimator y
ClassifierMixin, y tiene 20 pruebas unitarias (aprende en datos sintéticos con solución conocida, alcanza el
óptimo de SLSQP y de L-BFGS, casos límite, clone, entradas inválidas).

**¿Comparación justa?** Mismos pliegues, misma semilla, mismo presupuesto de TPE (24 evaluaciones), mismo
criterio (AUC interno). El combinador ve más filas etiquetadas (todas anteriores al corte): el control con 40
mil filas muestra que en k2-k4 sigue por encima de XGBoost; en el pliegue final, no.

**Revisión adversarial.** Contar que una revisión del código encontró que el optimizador de la versión 2.0 no
convergía y que el historial incluía semanas del hueco; ambas cosas se corrigieron en la v2.1 y se reportan
las iteraciones (v1, v2.0, v2.1) en la sección 5.9.

## Criterio 5 · Resultados y evaluación crítica (15 %)

Cifras para memorizar:

* AUC externo (media de 4 años): AJ-KL 0,7874 ± 0,0029; XGBoost 0,7867; LightGBM 0,7865; CatBoost 0,7860.
  El AJ-KL también tiene el mejor AUC-PR (0,5642), Brier (0,1613) y log-loss (0,4829).
* Prueba 2025 (validación anidada): AJ-KL 0,7985; XGBoost 0,7974; LightGBM 0,7981; CatBoost 0,7982.
* Producción (622 mil filas): AJ-KL 0,8007; XGBoost 0,7999 (ΔAUC +0,0008, p Holm < 0,001); LightGBM y
  CatBoost 0,8011 (−0,0004 para el AJ-KL, no significativo tras Holm).
* Frente a XGBoost rechazan cinco pruebas: Friedman (p < 0,001), Wilcoxon-Holm (p = 1,1 × 10⁻⁷; 39 de 48
  meses; d_z = 0,79), DeLong por bloques de semana (p = 3,5 × 10⁻⁹; ΔAUC +0,0011), MCS (solo queda el AJ-KL) y
  Diebold-Mariano (p = 8,9 × 10⁻⁸; 42 de 52 semanas). No rechazan Nemenyi (p = 0,74) ni Nadeau-Bengio (p = 0,13).

Puntos para la evaluación crítica: la ventaja es consistente pero trivial en unidades de AUC (+0,0007); el ECE
no mejora; frente al stacking logístico y al BLP con el mismo historial gana solo 0,0003 de AUC y 0,0001 de
log-loss, y esas reglas calibran mejor; el componente que más aporta es el encogimiento (sin él, −0,0017 de
AUC y +0,0097 de log-loss).

## Criterio 6 · Conclusiones y limitaciones (10 %)

* El techo de información domina: mejoras de milésimas.
* En producción (622 mil filas) la ventaja frente a XGBoost no es significativa y LightGBM/CatBoost
  reentrenados quedan levemente arriba: el combinador aprendió pesos con modelos de 40 mil filas.
* Iteraciones del diseño con resultados de 2025 ya conocidos: 2025 no es una prueba ciega del diseño.
* Trabajo futuro: reentrenar el combinador con predicciones de modelos del tamaño de producción, información
  anticipada de t+1 (lluvias, cierres viales) y umbral con costos explícitos.

## Criterio 7 · Dominio del tema (10 %)

Preguntas probables:

* **¿Qué pasa con un corredor nuevo?** Usa los pesos de su ancestro más profundo con pesos (central × grupo,
  central o raíz).
* **¿Por qué la calibración beta y no Platt?** Un pool lineal queda subdisperso (Ranjan y Gneiting, 2010); la
  calibración beta tiene dos pendientes (log q y −log(1−q)) y corrige ambas colas. Se activa solo si mejora la
  log-loss interna.
* **¿Qué significa la brecha de Frank-Wolfe?** Una cota superior del error en el objetivo que no depende del
  paso: si es menor que 10⁻⁸, el objetivo está a menos de 10⁻⁸ del óptimo.
* **¿Por qué tres pruebas?** La guía define superioridad como tres pruebas distintas con p ajustado < 0,05 y
  efecto no despreciable. Reportamos también las que no rechazan.
* **¿Dónde no gana?** Sección 5.12: subgrupos de distancia, escala y PDET, y centrales donde pierde frente a
  XGBoost.
