"""Bloque B: selección de la mejor combinación por modelo, reentrenamiento y serialización.

1. Para cada uno de los 7 modelos se elige la combinación (balanceo × optimizador) con mayor AUC
   medio en el bucle externo (k1-k4) y se verifica la pertenencia de la mejor global al grupo de
   finalistas de Friedman-Nemenyi del Entregable 2 (rango a menos de una diferencia crítica).
2. Cada tubería completa (preprocesamiento + balanceo + modelo) se reentrena con todo el periodo de
   entrenamiento (622.244 filas) usando los hiperparámetros elegidos en el pliegue final; k-NN se
   reentrena con la muestra de 40 mil filas del pliegue final (ver `METODO_KNN`).
3. Se serializa con joblib (compress=3) en artifacts/<modelo>/ con metadata.json y se comprueba
   que el modelo cargado reproduce exactamente las predicciones del modelo en memoria sobre una
   muestra fija de la prueba 2025 (prueba de ida y vuelta).

Uso: python scripts/13_artefactos.py [benchmarks|original|todos]
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import artefactos as A  # noqa: E402
import config as C  # noqa: E402
import datos as D  # noqa: E402
import evaluacion as EV  # noqa: E402
import experimento as X  # noqa: E402

ART = RAIZ / "artifacts"
RES = RAIZ / "data" / "resultados"
MODELOS = ["XGBoost", "Random Forest", "SVM", "Regresión logística", "KNN", "Decision Tree", "Naive Bayes"]
METRICAS = ["auc", "auc_pr", "f1", "precision", "recall", "accuracy", "brier", "ece", "log_loss"]
METODO_KNN = ("k-NN guarda todo el conjunto de referencia: con 622 mil filas y 70 variables el artefacto "
              "ocuparía unos 350 MB, por encima de la memoria del plan gratuito de Render, y su k = 301 se "
              "eligió para la densidad de la muestra de 40 mil filas. Se usa como referencia la muestra del "
              "pliegue final (40 mil filas), la misma con que se ajustaron sus hiperparámetros.")


def slug(nombre: str) -> str:
    tabla = str.maketrans("áéíóúñ ", "aeioun_")
    return nombre.lower().translate(tabla).replace("(", "").replace(")", "")


def seleccion(maestra: pd.DataFrame) -> pd.DataFrame:
    clf = maestra[maestra.tarea == "clf"]
    ext = clf[clf.pliegue != "final"]
    media = ext.groupby("combinacion")[METRICAS].mean()
    desv = ext.groupby("combinacion")[METRICAS].std()
    info = ext.groupby("combinacion")[["modelo", "balanceo", "optimizador"]].first()
    rangos = pd.read_parquet(RES / "comp_rangos_global.parquet")["rango"]
    cd = float(pd.read_parquet(RES / "comp_friedman_global.parquet")["CD"].iloc[0])
    filas = []
    for modelo in MODELOS:
        combos = info[info.modelo == modelo].index
        # Empates exactos en AUC (k-NN con y sin votos ponderados: la ponderación es monótona y no
        # cambia el orden) se resuelven con el Brier medio, que sí premia la calibración.
        orden = media.loc[combos].sort_values(["auc", "brier"], ascending=[False, True])
        mejor = orden.index[0]
        segundo = orden["auc"].iloc[1]
        fin = clf[(clf.combinacion == mejor) & (clf.pliegue == "final")].iloc[0]
        fila = {"modelo": modelo, "balanceo": info.loc[mejor, "balanceo"], "optimizador": info.loc[mejor, "optimizador"],
                "combinacion": mejor, "params_json": fin["params_json"], "umbral_final": float(fin["umbral"]),
                "rango_friedman": float(rangos[mejor]), "ventaja_sobre_segunda": float(media.loc[mejor, "auc"] - segundo)}
        for m in METRICAS:
            fila[f"{m}_media"] = float(media.loc[mejor, m])
            fila[f"{m}_de"] = float(desv.loc[mejor, m])
        filas.append(fila)
    sel = pd.DataFrame(filas).sort_values("auc_media", ascending=False).reset_index(drop=True)
    mejor_rango = float(rangos.min())
    sel["finalista_nemenyi"] = sel["rango_friedman"] - mejor_rango <= cd
    sel["mejor_global"] = sel.index == 0
    sel["motivo"] = [
        (f"Mayor AUC externo de sus 16 combinaciones ({r.auc_media:.4f} ± {r.auc_de:.4f}; "
         f"{r.ventaja_sobre_segunda:+.4f} sobre la segunda" + (", empate desempatado por Brier" if r.ventaja_sobre_segunda == 0 else "") + ")" + ("; mejor global y finalista de Friedman-Nemenyi"
                                                               if r.mejor_global else ""))
        for r in sel.itertuples()]
    return sel


def datos_entrenamiento():
    panel = D.cargar_panel()
    train, test = D.particion(panel)
    return train, test


def muestra_referencia(test: pd.DataFrame) -> pd.DataFrame:
    ruta = ART / "muestra_referencia.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)
    m = test.sample(2000, random_state=C.SEMILLA).sort_values("fila")
    m[["fila"] + C.X_COLS + [C.OBJ_CLF]].to_parquet(ruta, index=False)
    return m


def un_hilo(obj):
    """Fija n_jobs = 1 en todos los estimadores anidados. Con varios hilos, Random Forest suma las
    probabilidades de los árboles en un orden que cambia entre ejecuciones (diferencias de 1e-16) y la
    prueba de ida y vuelta exige predicciones idénticas; además, Render ofrece un solo núcleo."""
    vistos = set()

    def visitar(o):
        if id(o) in vistos or o is None:
            return
        vistos.add(id(o))
        if hasattr(o, "get_params") and "n_jobs" in o.get_params(deep=False):
            o.set_params(n_jobs=1)
        for atributo in ("steps", "named_steps", "estimador_", "estimador", "estimador_base", "miembros_", "knn_"):
            v = getattr(o, atributo, None)
            if isinstance(v, dict):
                v = list(v.values())
            if isinstance(v, list):
                for e in v:
                    visitar(e[1] if isinstance(e, tuple) else e)
            elif v is not None and not isinstance(v, (str, float, int)):
                visitar(v)
    visitar(obj)
    return obj


def ida_y_vuelta(modelo, ruta: Path, ref: pd.DataFrame) -> dict:
    un_hilo(modelo)
    p_mem = modelo.predict_proba(ref[C.X_COLS])[:, 1]
    joblib.dump(modelo, ruta, compress=3)
    cargado = joblib.load(ruta)
    p_carga = cargado.predict_proba(ref[C.X_COLS])[:, 1]
    np.save(ruta.parent / "pred_referencia.npy", p_mem)
    return {"identicas": bool(np.array_equal(p_mem, p_carga)), "max_dif_abs": float(np.max(np.abs(p_mem - p_carga))),
            "n": int(len(ref))}


def evaluar(modelo, test, umbral) -> tuple[dict, np.ndarray, float]:
    t0 = time.perf_counter()
    p = modelo.predict_proba(test[C.X_COLS])[:, 1]
    t_pred = time.perf_counter() - t0
    r = EV.metricas_clf(test[C.OBJ_CLF].to_numpy(), p, umbral, p)
    return {k: r[k] for k in METRICAS + ["umbral", "tn", "fp", "fn", "tp"]}, p, t_pred


def metadata(nombre, comb, params, est_params, train_X, n_filas, t_fit, t_pred, n_test, metricas_ext, metricas_test,
             rt, ruta, extra=None):
    return {"modelo": nombre, "combinacion": comb, "semilla": C.SEMILLA, "librerias": A.versiones(),
            "hiperparametros": params, "parametros_estimador": est_params, "filas_entrenamiento": int(n_filas),
            "periodo_entrenamiento": f"semanas objetivo hasta {C.FIN_TRAIN.date()}",
            "objetivo": f"{C.OBJ_CLF} = 1 si la razón de abastecimiento de t+1 frente a su promedio de 8 semanas es < 0,5",
            "esquema_entrada": A.esquema_entrada(train_X), "metricas_bucle_externo": metricas_ext,
            "metricas_prueba_2025": metricas_test, "segundos_entrenamiento": t_fit,
            "milisegundos_por_1000_predicciones": 1000 * t_pred / n_test * 1000,
            "prueba_ida_y_vuelta": rt, "archivo": ruta.name, "megabytes": ruta.stat().st_size / 2 ** 20,
            **(extra or {})}


def benchmarks(sel, train, test, ref, solo=None):
    filas, preds = [], {}
    for r in sel.itertuples():
        if solo and r.modelo not in solo:
            continue
        carpeta = ART / slug(r.modelo)
        carpeta.mkdir(parents=True, exist_ok=True)
        params = json.loads(r.params_json)
        datos = train
        extra = {}
        if r.modelo == "KNN":
            S, _ = X.cargar_pliegue("final")
            datos = train[train["fila"].isin(S["fila"])]
            extra["nota_entrenamiento"] = METODO_KNN
        y = datos[C.OBJ_CLF].to_numpy()
        if os.environ.get("REUSAR") == "1" and (carpeta / "modelo.joblib").exists():
            # artefacto ya serializado en una corrida anterior: se carga y se reevalúa sin reentrenar
            pipe = joblib.load(carpeta / "modelo.joblib")
            meta_previa = json.loads((carpeta / "metadata.json").read_text())
            umbral = meta_previa["metricas_prueba_2025"]["umbral"]
            m_test, p, t_pred = evaluar(pipe, test, umbral)
            preds[r.combinacion] = p.astype(np.float32)
            filas.append({"modelo": r.modelo, "combinacion": r.combinacion, "filas": meta_previa["filas_entrenamiento"],
                          "segundos_fit": meta_previa["segundos_entrenamiento"],
                          "ms_por_1000": meta_previa["milisegundos_por_1000_predicciones"], "mb": meta_previa["megabytes"],
                          "ida_y_vuelta": meta_previa["prueba_ida_y_vuelta"]["identicas"],
                          **{f"{k}_2025": v for k, v in m_test.items()}})
            print(filas[-1], flush=True)
            continue
        pipe = A.tuberia(r.modelo, r.balanceo, params, y)
        t0 = time.perf_counter()
        pipe.fit(datos[C.X_COLS], y)
        t_fit = time.perf_counter() - t0
        un_hilo(pipe)                      # la inferencia se mide y se sirve con un solo hilo
        umbral = r.umbral_final
        if r.modelo == "SVM":   # el umbral del Entregable 2 está en la escala del margen: se lleva a probabilidad
            umbral = float(pipe.named_steps["modelo"].platt_.predict(np.array([umbral]))[0])
        m_test, p, t_pred = evaluar(pipe, test, umbral)
        ruta = carpeta / "modelo.joblib"
        rt = ida_y_vuelta(pipe, ruta, ref)
        m_ext = {m: {"media": getattr(r, f"{m}_media"), "de": getattr(r, f"{m}_de")} for m in METRICAS}
        meta = metadata(r.modelo, r.combinacion, params, {k: str(v) for k, v in pipe.named_steps["modelo"].get_params().items()
                                                         if not hasattr(v, "get_params")},
                        datos[C.X_COLS], len(datos), t_fit, t_pred, len(test), m_ext, m_test, rt, ruta, extra)
        A.guardar_json(meta, carpeta / "metadata.json")
        preds[r.combinacion] = p.astype(np.float32)
        filas.append({"modelo": r.modelo, "combinacion": r.combinacion, "filas": len(datos), "segundos_fit": t_fit,
                      "ms_por_1000": meta["milisegundos_por_1000_predicciones"], "mb": meta["megabytes"],
                      "ida_y_vuelta": rt["identicas"], **{f"{k}_2025": v for k, v in m_test.items()}})
        print(filas[-1], flush=True)
    return pd.DataFrame(filas), preds


def original(train, test, ref):
    from sklearn.pipeline import Pipeline
    from ejul import EnsambleUmbralesLatentes
    import modelos as M
    res = joblib.load(RES / "original" / "final.joblib")["resumen"]
    ext = [joblib.load(RES / "original" / f"{pl}.joblib")["resumen"] for pl in ["k1", "k2", "k3", "k4"]]
    base = M.construir(M.catalogo("clf")["XGBoost"], res["params_base"])
    base.set_params(n_jobs=8)
    pipe = Pipeline([("prep", A.prep_ejul()),
                     ("modelo", EnsambleUmbralesLatentes(estimador_base=base, columnas_grupo=(-3, -2, -1),
                                                         random_state=C.SEMILLA, codificacion=res.get("codificacion", "cruzada"),
                                                         **res["params"]))])
    y = train[C.OBJ_CLF].to_numpy()
    t0 = time.perf_counter()
    pipe.fit(train[C.X_COLS], y, modelo__latente=train["ratio_t1"].to_numpy(), modelo__tiempos=train["semana_t1"].to_numpy())
    t_fit = time.perf_counter() - t0
    un_hilo(pipe)
    m_test, p, t_pred = evaluar(pipe, test, res["umbral"])
    carpeta = ART / "ejul_original"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "modelo.joblib"
    rt = ida_y_vuelta(pipe, ruta, ref)
    m_ext = {m: {"media": float(np.mean([e[m] for e in ext])), "de": float(np.std([e[m] for e in ext], ddof=1))}
             for m in METRICAS}
    meta = metadata("EJUL (original)", "EJUL (original) | Sin balanceo | Bayesiana (TPE)", res["params"],
                    {"params_base_xgboost": res["params_base"]}, train[C.X_COLS], len(train), t_fit, t_pred, len(test),
                    m_ext, m_test, rt, ruta,
                    {"pesos_umbrales": pipe.named_steps["modelo"].pesos_.tolist(),
                     "umbrales_latentes": pipe.named_steps["modelo"].umbrales_.tolist()})
    A.guardar_json(meta, carpeta / "metadata.json")
    fila = {"modelo": "EJUL (original)", "combinacion": meta["combinacion"], "filas": len(train), "segundos_fit": t_fit,
            "ms_por_1000": meta["milisegundos_por_1000_predicciones"], "mb": meta["megabytes"],
            "ida_y_vuelta": rt["identicas"], **{f"{k}_2025": v for k, v in m_test.items()}}
    print(fila, flush=True)
    return pd.DataFrame([fila]), {meta["combinacion"]: p.astype(np.float32)}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    C.fijar_semillas(C.SEMILLA)
    que = sys.argv[1] if len(sys.argv) > 1 else "todos"
    solo = sys.argv[2].split(",") if len(sys.argv) > 2 else None   # p. ej. "Random Forest,SVM"
    maestra = pd.read_parquet(RES / "maestra.parquet")
    sel = seleccion(maestra)
    sel.to_csv(RES / "seleccion_mejores.csv", index=False)
    print(sel[["modelo", "balanceo", "optimizador", "auc_media", "auc_de", "rango_friedman", "finalista_nemenyi"]].to_string())
    train, test = datos_entrenamiento()
    ref = muestra_referencia(test)
    tablas, preds = [], {}
    if que in ("benchmarks", "todos"):
        t, p = benchmarks(sel, train, test, ref, solo)
        tablas.append(t); preds.update(p)
    if que in ("original", "todos"):
        t, p = original(train, test, ref)
        tablas.append(t); preds.update(p)
    ruta_t = RES / "artefactos_resumen.csv"
    previo = pd.read_csv(ruta_t) if ruta_t.exists() else pd.DataFrame()
    nuevo = pd.concat([previo[~previo["combinacion"].isin(pd.concat(tablas)["combinacion"])] if len(previo) else previo]
                      + tablas, ignore_index=True)
    nuevo.to_csv(ruta_t, index=False)
    ruta_p = RES / "pred_prueba_artefactos.parquet"
    base = pd.read_parquet(ruta_p) if ruta_p.exists() else test[["fila", "semana_t1", C.OBJ_CLF]].reset_index(drop=True)
    for k, v in preds.items():
        base[k] = v
    base.to_parquet(ruta_p, index=False)
