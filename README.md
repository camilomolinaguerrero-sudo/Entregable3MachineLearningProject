# Entregable 3 · Interrupciones en corredores de abastecimiento (SIPSA)

Proyecto final de Machine Learning, Doctorado en Ingeniería. Camilo Molina Guerrero y Lina Margarita Buelvas.

* **Informe (Jupyter Book):** <https://camilomolinaguerrero-sudo.github.io/Entregable3MachineLearningProject/>
* **Dashboard (Render):** <https://sipsa-corredores-e3.onrender.com>

¿Se puede anticipar, con la información disponible al cierre de una semana, qué corredores municipio → central
mayorista tendrán la semana siguiente menos de la mitad de su abastecimiento habitual? El repositorio contiene
el panel de 744.921 corredor-semanas (SIPSA-A, DANE, 2018-2025), los artefactos de los siete benchmarks del
curso, LightGBM, CatBoost y el modelo original **AJ-KL** (apilamiento jerárquico con encogimiento KL y olvido
temporal, `src/original_model.py`), el dashboard en Dash y el libro.

## Estructura

| Carpeta | Contenido |
|---|---|
| `app.py`, `pages/`, `assets/` | Dashboard multipágina (Contexto, EDA, Modelos) |
| `src/` | Modelo original, carga de datos, figuras, artefactos, explicación SHAP |
| `scripts/` | Pipeline numerado (ver capítulo 9 del libro) |
| `artifacts/` | 10 modelos serializados con `metadata.json` y prueba de ida y vuelta |
| `data/` | Panel, resultados de la validación anidada, revisión de literatura, datos del dashboard |
| `cuadernos/`, `book/` | Fuentes y Jupyter Book |
| `tests/` | Pruebas unitarias (`pytest tests/`) |

## Ejecutar

```bash
pip install -r requirements.txt
python app.py
```

El entorno completo de los experimentos está en `requirements-dev.txt`; el orden de ejecución, en el capítulo 9.
