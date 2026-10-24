"""Artefactos de producción de LightGBM, CatBoost y del modelo original AJ-KL.

1. LightGBM y CatBoost (comparadores del estado del arte y modelos base del AJ-KL) se reentrenan con las
   622.244 filas de entrenamiento con los hiperparámetros que eligió su bucle interno en el pliegue final.
2. El AJ-KL de producción combina los 9 modelos base de producción con el combinador del pliegue final
   (entrenado con las predicciones fuera de muestra de los bloques externos 2020-2024). Los diccionarios
   de la jerarquía son los mismos del entrenamiento del combinador.
3. Cada artefacto se serializa con joblib (compress=3), con metadata.json y prueba de ida y vuelta.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import artefactos as A  # noqa: E402
import config as C  # noqa: E402
import datos as D  # noqa: E402
import preprocesamiento as P  # noqa: E402

RES = RAIZ / "data" / "resultados"
ART = RAIZ / "artifacts"
METR = ["auc", "auc_pr", "f1", "precision", "recall", "accuracy", "brier", "ece", "log_loss"]


def modulo(nombre, ruta):
    import importlib.util
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    warnings.filterwarnings("ignore")
    m13 = modulo("m13", RAIZ / "scripts" / "13_artefactos.py")
    m22 = modulo("m22", RAIZ / "scripts" / "22_ajkl.py")
    panel = D.cargar_panel()
    train, test = D.particion(panel)
    ref = pd.read_parquet(ART / "muestra_referencia.parquet")
    y = train[C.OBJ_CLF].to_numpy()
    filas, preds = [], {}
    # ------------------------------------------------------------------ LightGBM y CatBoost
    for nombre in m22.SOTA:
        r = joblib.load(RES / "sota" / f"{nombre}_final.joblib")["resumen"]
        ext = [joblib.load(RES / "sota" / f"{nombre}_{pl}.joblib")["resumen"] for pl in ["k1", "k2", "k3", "k4"]]
        carpeta = ART / nombre.lower()
        if (carpeta / "metadata.json").exists():          # ya entrenado en una corrida anterior
            pipe = joblib.load(carpeta / "modelo.joblib")
            t_fit = json.loads((carpeta / "metadata.json").read_text())["segundos_entrenamiento"]
        else:
            pipe = Pipeline([("prep", P.preprocesador("arboles", "clf")), ("modelo", m22.construir_sota(nombre, r["params"], n_jobs=8))])
            t0 = time.perf_counter()
            pipe.fit(train[C.X_COLS], y)
            t_fit = time.perf_counter() - t0
            if nombre == "LightGBM":
                m13.un_hilo(pipe)
        m_test, p, t_pred = m13.evaluar(pipe, test, r["umbral"])
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / "modelo.joblib"
        rt = m13.ida_y_vuelta(pipe, ruta, ref)
        m_ext = {k: {"media": float(np.mean([e[k] for e in ext])), "de": float(np.std([e[k] for e in ext], ddof=1))} for k in METR}
        meta = m13.metadata(nombre, f"{nombre} | Sin balanceo | Bayesiana (TPE)", r["params"], {}, train[C.X_COLS], len(train),
                            t_fit, t_pred, len(test), m_ext, m_test, rt, ruta,
                            {"nota": "Comparador reciente del estado del arte (no visto en clase) y modelo base del AJ-KL."})
        A.guardar_json(meta, carpeta / "metadata.json")
        preds[meta["combinacion"]] = p.astype(np.float32)
        filas.append({"modelo": nombre, "combinacion": meta["combinacion"], "filas": len(train), "segundos_fit": t_fit,
                      "ms_por_1000": meta["milisegundos_por_1000_predicciones"], "mb": meta["megabytes"],
                      "ida_y_vuelta": rt["identicas"], **{f"{k}_2025": v for k, v in m_test.items()}})
        print(filas[-1], flush=True)
    # ------------------------------------------------------------------ AJ-KL de producción
    sel = m22.seleccion()
    orden = m22.columnas_base(sel)
    carpetas = {c: m13.slug(c.split(" | ")[0]) for c in sel["combinacion"]}
    carpetas.update({m: m.lower() for m in m22.SOTA})
    bases = {c: joblib.load(ART / carpetas[c] / "modelo.joblib") for c in orden}
    fin = joblib.load(RES / "ajkl" / "final.joblib")
    ext = [joblib.load(RES / "ajkl" / f"{pl}.joblib")["resumen"] for pl in ["k1", "k2", "k3", "k4"]]
    c = panel["mercado"].astype(str)
    cg = c + "|" + panel["grupo_dom"].astype(str)
    mapas = []
    for clave in [c, cg, cg + "|" + panel["cod_mpio"].astype(str)]:
        codigos, unicos = pd.factorize(clave)
        mapas.append({u: i for i, u in enumerate(unicos)})
    ajkl = A.AJKLProduccion(bases=bases, combinador=fin["modelo"], mapas=mapas, orden=orden).fit()
    m_test, p, t_pred = m13.evaluar(ajkl, test, fin["resumen"]["umbral"])
    carpeta = ART / "ajkl_original"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "modelo.joblib"
    rt = m13.ida_y_vuelta(ajkl, ruta, ref)
    m_ext = {k: {"media": float(np.mean([e[k] for e in ext])), "de": float(np.std([e[k] for e in ext], ddof=1))} for k in METR}
    meta = m13.metadata("AJ-KL (original)", "AJ-KL (original) | Sin balanceo | Bayesiana (TPE)", fin["resumen"]["params"],
                        {"modelos_base": orden, "nodos_por_nivel": [len(d) for d in fin["modelo"].pesos_],
                         "pesos_raiz": dict(zip([o.split(" | ")[0] for o in orden], map(float, fin["modelo"].raiz_)))},
                        train[C.X_COLS], len(train), 0.0, t_pred, len(test), m_ext, m_test, rt, ruta,
                        {"nota": ("Los modelos base son los artefactos de producción (622 mil filas). El combinador es el del "
                                  "pliegue final de la validación anidada, entrenado con las predicciones fuera de muestra de "
                                  "los bloques externos de 2020 a 2024.")})
    A.guardar_json(meta, carpeta / "metadata.json")
    preds[meta["combinacion"]] = p.astype(np.float32)
    filas.append({"modelo": "AJ-KL (original)", "combinacion": meta["combinacion"], "filas": len(train), "segundos_fit": 0.0,
                  "ms_por_1000": meta["milisegundos_por_1000_predicciones"], "mb": meta["megabytes"],
                  "ida_y_vuelta": rt["identicas"], **{f"{k}_2025": v for k, v in m_test.items()}})
    print(filas[-1], flush=True)
    # ------------------------------------------------------------------ tablas compartidas
    ruta_t = RES / "artefactos_resumen.csv"
    previo = pd.read_csv(ruta_t)
    previo = previo[~previo["modelo"].isin(["EJUL (original)"] + [f["modelo"] for f in filas])]
    pd.concat([previo, pd.DataFrame(filas)], ignore_index=True).to_csv(ruta_t, index=False)
    ruta_p = RES / "pred_prueba_artefactos.parquet"
    base = pd.read_parquet(ruta_p)
    base = base.drop(columns=[k for k in base.columns if k.startswith("EJUL")])
    for k, v in preds.items():
        base[k] = v
    base.to_parquet(ruta_p, index=False)


if __name__ == "__main__":
    main()
