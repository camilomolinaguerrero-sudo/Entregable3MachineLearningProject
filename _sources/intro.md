# Dashboard predictivo, modelo original e informe reproducible

**Tercer entregable del proyecto final de Machine Learning** · Doctorado en Ingeniería

**Autores:** Camilo Molina Guerrero y Lina Margarita Buelvas

---

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} 📊 Dashboard
:link: https://sipsa-corredores-e3.onrender.com
Contexto, EDA y modelos, con predicción en vivo
:::

:::{grid-item-card} 💻 Código
:link: https://github.com/camilomolinaguerrero-sudo/Entregable3MachineLearningProject
Repositorio con scripts, módulos, artefactos y este libro
:::

:::{grid-item-card} 📘 Entregables anteriores
:link: https://camilomolinaguerrero-sudo.github.io/Entregable2MachineLearningProject/
Entregable 2: 140 modelos con validación anidada
:::
::::

## Resumen ejecutivo

Con los microdatos del componente de abastecimiento del SIPSA (DANE) se construyó un panel semanal de
744.921 corredor-semanas: 4.512 corredores entre 925 municipios de origen y las 32 principales centrales
mayoristas del país, de 2018 a 2025. La pregunta es si se puede anticipar, con la información disponible al
cierre de una semana, qué corredores tendrán la semana siguiente un abastecimiento menor que la mitad de su
promedio de 8 semanas (una **interrupción**, el 27,9 % de las corredor-semanas).

Este entregable cierra el proyecto con cuatro productos:

1. **Correcciones del Entregable 2**, con una tabla de trazabilidad (capítulo 1).
2. **Selección, reentrenamiento y serialización** de la mejor combinación de cada uno de los siete modelos del
   Entregable 2, más dos comparadores recientes (LightGBM y CatBoost), con `metadata.json` y prueba de ida y
   vuelta (capítulo 3).
3. **Un modelo original**, el Apilamiento Jerárquico con encogimiento KL y olvido temporal (**AJ-KL**),
   programado con numpy y scipy, con fundamento teórico, pruebas unitarias, ablación, evaluación en tres
   conjuntos de datos adicionales y validación estadística (capítulos 4 a 6).
4. **Un dashboard multipágina en Dash**, desplegado en Render, que carga los modelos ya entrenados (capítulo 7).

## Resultados principales

| Aspecto | Resultado |
|---|---|
| Mejor benchmark del curso | XGBoost sin balanceo con Random Search: AUC externo 0,7867 ± 0,0027 |
| Idea del modelo original | Ningún modelo es el mejor en todas partes. El AJ-KL combina las probabilidades de 9 modelos (los 7 del curso, LightGBM y CatBoost) con pesos distintos para cada nodo de la jerarquía país → central → central × grupo de alimentos → corredor. Los pesos de cada nodo minimizan la log-loss con más peso para las semanas recientes y se encogen hacia los de su nodo padre con una penalización KL, resuelta con descenso espejo hasta un certificado de optimalidad |
| Novedad | Búsqueda en Web of Science, Scopus y ScienceDirect (683 registros, 230 revisados) y 20 cadenas adicionales en Web of Science: ningún trabajo aplica una combinación local o jerárquica de clasificadores a interrupciones de corredores de abastecimiento. Lo más cercano por método es el apilamiento jerárquico bayesiano (Yao et al., 2022) |
| Validación anidada (bucle externo 2020-2024) | AUC 0,7874 ± 0,0029, el mayor de todos; también el mejor AUC-PR, Brier y log-loss externos, y el mejor F1 entre los modelos del curso (CatBoost lo iguala) |
| Superioridad frente a XGBoost (R3) | Cinco pruebas rechazan la igualdad con p ajustado < 0,05: Friedman con permutación por pliegue, Wilcoxon-Holm (39 de 48 meses a favor, d de Cohen pareada 0,79), DeLong con bloques de semana en 2025 (ΔAUC +0,0011, p Holm < 0,001), el Model Confidence Set (solo el AJ-KL queda en el conjunto, en Brier y en log-loss) y Diebold-Mariano sobre la log-loss semanal de 2025 (42 de 52 semanas). No rechazan Nemenyi ni Nadeau-Bengio |
| Estado del arte reciente | Gana a LightGBM en 40 y a CatBoost en 44 de los 48 meses del bucle externo (Wilcoxon-Holm p < 0,001); en 2025 queda por encima (+0,0004 y +0,0003) pero sin significancia tras Holm |
| Modelos de producción (622 mil filas, 2025) | AJ-KL 0,8007 frente a 0,7999 de XGBoost (ΔAUC +0,0008, IC BCa [+0,0004; +0,0012], p Holm < 0,001) y la mejor log-loss; LightGBM y CatBoost reentrenados llegan a 0,8011 (diferencia no significativa tras Holm) |
| Conjuntos adicionales | Mayor AUC medio que XGBoost en los tres (Beijing PM2.5, Bike Sharing y Bank Marketing) |
| Límites | La ventaja es trivial en unidades de AUC (+0,0007 en el bucle externo) y cae dentro de la región de equivalencia práctica de ±0,01; frente al stacking logístico y al BLP con el mismo historial gana por 0,0003 de AUC y 0,0001 de log-loss, pero su ECE es peor (≈ 0,001); casi toda la ganancia la dan la raíz y la central |

## Cómo leer este libro

0. **Introducción** (esta página).
1. **Trazabilidad de correcciones** del Entregable 2.
2. **Datos: ETL y EDA**.
3. **Modelos benchmark**: selección estadística, reentrenamiento, artefactos y comparadores del estado del arte.
4. **Revisión de literatura y novedad** en Web of Science, Scopus y ScienceDirect.
5. **Modelo original**: formulación, propiedades con demostración, pseudocódigo, complejidad, implementación,
   pruebas unitarias, convergencia, ablación, iteraciones del diseño, interpretabilidad y conjuntos adicionales.
6. **Validación estadística**: protocolo, supuestos, pruebas, ajustes por comparaciones múltiples, intervalos,
   tamaños del efecto, estado del arte y modelos de producción.
7. **Dashboard y despliegue**.
8. **Conclusiones**, veredicto sobre R1, R2 y R3, limitaciones y declaración de uso de herramientas de IA.
9. **Reproducibilidad y bibliografía**.
