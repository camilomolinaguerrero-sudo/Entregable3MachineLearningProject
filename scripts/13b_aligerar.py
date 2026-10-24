"""Aligera los artefactos sin cambiar sus predicciones (guía, sección 4.2: tamaño de los artefactos).

* EJUL: elimina los arreglos de entrenamiento que solo sirven para ajustar la recalibración y para la
  ablación (predicciones fuera de pliegue de las 622 mil filas) y los miembros de los pliegues cuando la
  predicción usa los miembros completos.
* Tuberías con SMOTE o ADASYN: el remuestreo solo actúa en `fit`; se reemplaza el paso ajustado (que
  guarda los vecinos de la clase minoritaria) por una copia sin ajustar.
Después se repite la prueba de ida y vuelta contra las predicciones de referencia guardadas y se
actualiza el tamaño en metadata.json.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config as C  # noqa: E402

ART = RAIZ / "artifacts"


def aligerar(pipe):
    modelo = pipe.named_steps.get("modelo")
    for atributo in ("oof_", "oof_crudo_", "oof_miembros_"):
        if hasattr(modelo, atributo):
            delattr(modelo, atributo)
    if getattr(modelo, "prediccion", None) == "completo" and hasattr(modelo, "miembros_pliegue_"):
        delattr(modelo, "miembros_pliegue_")
    if "balanceo" in pipe.named_steps:
        i = [n for n, _ in pipe.steps].index("balanceo")
        pipe.steps[i] = ("balanceo", clone(pipe.steps[i][1]))
    return pipe


def main():
    ref = pd.read_parquet(ART / "muestra_referencia.parquet")
    for carpeta in sorted(p for p in ART.iterdir() if (p / "modelo.joblib").exists()):
        ruta = carpeta / "modelo.joblib"
        antes = ruta.stat().st_size / 2 ** 20
        pipe = aligerar(joblib.load(ruta))
        joblib.dump(pipe, ruta, compress=3)
        p = joblib.load(ruta).predict_proba(ref[C.X_COLS])[:, 1]
        esperado = np.load(carpeta / "pred_referencia.npy")
        identicas = bool(np.array_equal(p, esperado))
        meta = json.loads((carpeta / "metadata.json").read_text())
        meta["megabytes"] = ruta.stat().st_size / 2 ** 20
        meta["prueba_ida_y_vuelta"] = {"identicas": identicas, "max_dif_abs": float(np.max(np.abs(p - esperado))),
                                       "n": int(len(ref)), "nota": "repetida después de aligerar el artefacto"}
        (carpeta / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"{carpeta.name:22s} {antes:7.2f} MB -> {meta['megabytes']:7.2f} MB  idénticas: {identicas}")
        assert identicas, carpeta.name
    resumen = pd.read_csv(RAIZ / "data" / "resultados" / "artefactos_resumen.csv")
    resumen["mb"] = [json.loads((ART / s / "metadata.json").read_text())["megabytes"]
                     for s in ["xgboost", "random_forest", "svm", "regresion_logistica", "knn", "decision_tree", "naive_bayes", "ejul_original"]]
    resumen.to_csv(RAIZ / "data" / "resultados" / "artefactos_resumen.csv", index=False)


if __name__ == "__main__":
    main()
