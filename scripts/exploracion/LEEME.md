# Scripts exploratorios (diseño del modelo original)

Pruebas rápidas que guiaron el diseño del EJUL antes del experimento anidado formal. Se conservan para
la trazabilidad de las iteraciones descritas en la sección 5.8 del libro. Escriben sus resultados en la
carpeta temporal indicada en la línea de comandos; las copias usadas en el libro están en
`data/resultados/iteraciones/`.

| Script | Pregunta |
|---|---|
| prueba_umbrales.py, prueba_corredor.py, prueba_estructura.py, prueba_combo.py | ¿Aportan los umbrales vecinos y la codificación del corredor sobre XGBoost? |
| prueba_control.py | ¿Cuánto de la mejora se debe solo a promediar semillas? |
| prueba_recencia.py, prueba_suave.py | Ponderación por recencia y etiquetas suaves (no ayudaron) |
| valida_ejul.py, diag_calib2.py | Primera versión de la clase y variantes de calibración en k1-k4 |
| prueba_pliegues.py | Predicción con los modelos de los pliegues de recalibración |
| prueba_escala.py | Codificación cruzada frente a temporal con todos los datos de k3 y k4 |
