# 8. Conclusiones y limitaciones

## 8.1 Hallazgos

1. **El techo de información sigue siendo el principal límite.** Los siete modelos del Entregable 2 quedan
   dentro de 0,02 de AUC entre sí y ninguna variable observada al cierre de la semana se asocia con fuerza con
   el abastecimiento de la semana siguiente (la mayor correlación de Spearman en valor absoluto es 0,20). Las
   mejoras posibles con más modelado son pequeñas, y el AJ-KL lo confirma: combina los nueve mejores modelos
   disponibles y gana por menos de una milésima de AUC.
2. **Combinar con pesos locales encogidos supera de forma consistente al mejor modelo individual.** El AJ-KL
   tiene el mayor AUC externo medio (0,7874 frente a 0,7867 de XGBoost) y también el mejor AUC-PR, Brier
   y log-loss; gana a XGBoost en 39 de los 48 meses del bucle externo, en los cuatro años y en la prueba 2025
   de la validación anidada (+0,0011, DeLong con bloques de semana p Holm < 0,001, y 42 de 52 semanas con menor
   log-loss). Con los modelos de producción la ventaja frente a XGBoost se mantiene (+0,0008, p Holm < 0,001).
   En los tres conjuntos de datos adicionales tiene mayor AUC medio que XGBoost.
3. **El encogimiento es el componente que más aporta; la profundidad, poco.** Quitar el encogimiento (λ = 0)
   empeora la log-loss en los cinco pliegues y el ECE en cuatro (ΔAUC −0,0017, ΔAUC-PR −0,011), porque los
   nodos pequeños sobreajustan. La profundidad
   elegida por TPE fue 4 (hasta el corredor) en tres de los cinco pliegues, pero la diferencia frente a
   profundidades menores es de diezmilésimas: casi toda la ganancia la dan los pesos de la raíz y de la central.
   Los controles de la sección 5.8 muestran que la ganancia no viene de incluir LightGBM y CatBoost como base
   (con solo los 7 modelos del curso el AUC baja apenas 0,0002) ni de ver más filas etiquetadas que los
   benchmarks (con el historial reducido a 40 mil filas el AJ-KL sigue por encima de XGBoost en los cuatro
   pliegues donde el historial era mayor). Frente al stacking logístico y al BLP ajustados con el mismo
   historial, el AJ-KL gana por 0,0003 de AUC y 0,0001 de log-loss, pero su ECE es algo peor.
4. **Un optimizador que se detiene antes del óptimo es una regularización oculta.** La revisión adversarial
   del código mostró que la primera implementación, con un número fijo de iteraciones, dejaba los pesos lejos
   del minimizador y que la teoría no describía el estimador evaluado. Resolver cada nodo hasta la brecha de
   Frank-Wolfe hizo que λ recuperara su papel y que los pesos y la ablación se puedan interpretar.

## 8.2 Veredicto sobre las condiciones del modelo original

* **R1, novedad en la aplicación: se cumple** según la búsqueda en Web of Science, Scopus y ScienceDirect del
  capítulo 4, complementada con 20 cadenas por aplicación en Web of Science. Ningún registro aplica una
  combinación local o jerárquica de clasificadores a la predicción de interrupciones de corredores de
  abastecimiento. El trabajo más cercano por método, el apilamiento jerárquico bayesiano
  {cite}`yao2022hierarchical`, usa MCMC con pesos softmax-lineales y no tiene olvido; el más cercano por
  aplicación {cite}`nino2026disruption` es causal, no predictivo.
* **R2, implementación algorítmica propia: se cumple.** La clase `ApilamientoJerarquicoKL` hereda de
  `BaseEstimator` y `ClassifierMixin`; el objetivo, el gradiente, la actualización de descenso espejo, la
  brecha de Frank-Wolfe, el recorrido de la jerarquía y la calibración beta por Newton ponderado están
  programados con numpy y scipy, con pseudocódigo, seis proposiciones demostradas, complejidad contrastada con
  tiempos medidos y 20 pruebas unitarias del modelo. Los modelos base son los artefactos del capítulo 3; el
  combinador no proviene de ningún `import`.
* **R3, superioridad estadística: se cumple frente a los benchmarks del curso con la definición operativa de
  la guía, con matices.** (1) El AUC externo medio del AJ-KL (0,7874) supera al de las siete mejores
  combinaciones. (2) Cinco pruebas distintas rechazan la igualdad frente a XGBoost con p ajustado < 0,05 (0,10
  en el MCS): Friedman con permutación por pliegue, Wilcoxon-Holm sobre 48 meses, DeLong con error estándar
  por bloques de semana en 2025, el Model Confidence Set (que en Brier y en log-loss semanales conserva solo al
  AJ-KL) y Diebold-Mariano sobre la log-loss semanal de 2025. (3) El tamaño del efecto no es despreciable: d de
  Cohen pareada 0,79 y dominancia pareada 0,62 (grande). Los matices: no rechazan Nemenyi (conservadora con 8
  modelos) ni la t de Nadeau-Bengio con cinco pliegues; la diferencia es trivial en unidades de AUC (+0,0007) y
  cae dentro de la región de equivalencia práctica de ±0,01; y el delta de Cliff de dos muestras es
  despreciable (0,05), porque la variación entre meses es mucho mayor que la diferencia entre modelos.

## 8.3 Limitaciones

* **Modelos de producción y comparadores recientes.** Con los nueve modelos base reentrenados con las 622 mil
  filas, el AJ-KL obtiene en 2025 un AUC de 0,8007, por encima de XGBoost (0,7999, p Holm < 0,001) y con la
  mejor log-loss, pero LightGBM y CatBoost reentrenados llegan a 0,8011 (diferencia de −0,0004, no
  significativa tras Holm). Una explicación probable es un desajuste de régimen: el combinador aprendió los
  pesos con predicciones de modelos entrenados con 40 mil filas y en producción combina modelos más fuertes y
  más parecidos entre sí. Reentrenar el combinador con predicciones fuera de muestra de modelos de 622 mil
  filas exige repetir la validación temporal completa de los nueve modelos y queda como trabajo futuro. En el
  bucle externo el AJ-KL gana a LightGBM en 40 y a CatBoost en 44 de los 48 meses (Wilcoxon-Holm p < 0,001), y
  en la prueba 2025 de la validación anidada queda por encima de ambos sin significancia tras Holm.
* **Calibración.** El ECE externo medio del AJ-KL (0,0117) es parecido al de XGBoost (0,0115) y peor que el de
  Random Forest (0,0086) y el del stacking logístico con el mismo historial.
* **Conjuntos de datos adicionales.** Solo se evaluaron cuatro pliegues por conjunto, insuficientes para una
  prueba formal por conjunto; en Bank Marketing, con un cambio de régimen fuerte entre años, el AJ-KL gana a
  XGBoost en 2 de 4 pliegues y el mejor modelo es la regresión logística.
* **Iteraciones del diseño.** El modelo final es la cuarta versión (EJUL exploratorio, AJ-KL v1, v2.0 y v2.1) y
  los resultados de 2025 de las anteriores se conocían. Los cambios de la última versión se hicieron por
  corrección, no por desempeño, pero 2025 ya no es una prueba ciega para el diseño.
* **Información del combinador.** En los pliegues k2 a final el combinador ve las predicciones de los bloques
  externos anteriores (entre 76 y 375 mil filas etiquetadas, todas anteriores al corte de entrenamiento), más
  que la muestra de 40 mil filas de los benchmarks. El control con 40 mil filas conserva la ventaja sobre
  XGBoost, pero la reduce (ΔAUC medio −0,0007 frente al modelo completo).
* **SMOTE sobre variables one-hot** en la combinación seleccionada de Naive Bayes (heredado del Entregable 2).

## 8.4 Trabajo futuro

Reentrenar el combinador con predicciones de modelos del mismo tamaño que los de producción; incorporar
información anticipada de la semana *t+1* (pronósticos de lluvias, alertas de cierres viales, como los que
estudia {cite:t}`nino2026disruption`); y elegir el umbral de alerta con un criterio de costo explícito en lugar
del F1.

## 8.5 Declaración de autoría y uso de herramientas

El trabajo es de los dos autores, que responden por cada línea entregada y pueden explicarla y modificarla en
la sustentación. Se usaron como herramientas: Python y las librerías listadas en `requirements-dev.txt`; Web
of Science, Scopus y ScienceDirect (acceso institucional) para la revisión de literatura; y un asistente de IA generativa (Claude, de
Anthropic) como apoyo en la escritura y depuración de código, la automatización de las búsquedas y el cribado
inicial de registros, la redacción de borradores del texto, la revisión adversarial del código y de las demostraciones
(que encontró el problema de convergencia corregido en la versión 2.1) y la revisión de consistencia entre
cifras y resultados. Las decisiones de diseño del modelo, la interpretación de los resultados y las conclusiones fueron
revisadas por los autores, que verificaron los resultados contra las salidas de los scripts.

| Integrante | Firma |
|---|---|
| Camilo Molina Guerrero | ____________________ |
| Lina Margarita Buelvas | ____________________ |
