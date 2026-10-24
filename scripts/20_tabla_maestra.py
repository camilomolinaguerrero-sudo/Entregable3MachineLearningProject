"""Tabla maestra de experimentos (guía, sección 10, entregable 5).

Une las 140 combinaciones del Entregable 2 (clasificación y regresión, 5 pliegues cada una), el modelo
original (AJ-KL) y los comparadores del estado del arte (LightGBM y CatBoost), con las métricas de cada
pliegue externo y de la prueba 2025, los tiempos y la semilla.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "data" / "resultados"
sys.path.insert(0, str(RAIZ / "src"))      # las clases del modelo original se deserializan desde src/


def filas(carpeta: Path, patron: str, grupo: str):
    salida = []
    for f in sorted(carpeta.glob(patron)):
        r = dict(joblib.load(f)["resumen"])
        r["params_json"] = json.dumps(r.pop("params"), ensure_ascii=False)
        for k in ["params_base", "pesos", "umbrales_latentes", "calibracion", "modelos_base", "nodos"]:
            if k in r:
                r[k] = json.dumps(r[k])
        r["grupo"] = grupo
        r["combinacion"] = f"{r['modelo']} | {r['balanceo']} | {r['optimizador']}"
        salida.append(r)
    return salida


def main():
    e2 = pd.read_parquet(RES / "maestra.parquet").assign(grupo="Entregable 2 (140 combinaciones)")
    extra = filas(RES / "ajkl", "*.joblib", "Modelo original (AJ-KL v2.1)")
    extra += filas(RES / "sota", "*.joblib", "Estado del arte (LightGBM, CatBoost)")
    tabla = pd.concat([e2, pd.DataFrame(extra)], ignore_index=True)
    primeras = ["grupo", "tarea", "modelo", "balanceo", "optimizador", "combinacion", "pliegue", "semilla", "presupuesto",
                "evaluaciones", "auc", "auc_pr", "accuracy", "precision", "recall", "f1", "brier", "ece", "log_loss",
                "umbral", "rmse", "mae", "r2", "t_busqueda", "t_refit", "t_pred", "t_total", "filas_entrenamiento",
                "filas_eval", "params_json"]
    tabla = tabla[[c for c in primeras if c in tabla.columns] + [c for c in tabla.columns if c not in primeras]]
    tabla.to_csv(RES / "tabla_maestra.csv", index=False)
    tabla.to_parquet(RES / "tabla_maestra.parquet", index=False)
    print(tabla.groupby("grupo").size())


if __name__ == "__main__":
    main()
