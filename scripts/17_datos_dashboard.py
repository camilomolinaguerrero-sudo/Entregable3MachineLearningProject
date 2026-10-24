"""Datos livianos para el dashboard (data/dashboard/). El servidor solo lee estos archivos y los
artefactos de artifacts/; no entrena ni optimiza nada.

Parte EDA (no depende de los modelos):
  eda_muestra.parquet     muestra estratificada por año (80 mil corredor-semanas) con las variables
                          del modelo, el objetivo y la razón de abastecimiento de t+1
  eda_semanal.parquet     corredores e interrupciones por semana × central × departamento (panel completo)
  eda_faltantes.parquet   porcentaje de faltantes por variable y año (panel completo)
  eda_atipicos.parquet    resumen de atípicos (regla de Tukey) por variable numérica
  diccionario.csv         nombre, descripción, tipo y rango de cada variable
  resumen_eda.json        cifras clave
Parte modelos (requiere scripts 12-15): ver `modelos()`.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config as C  # noqa: E402
import datos as D  # noqa: E402

SAL = RAIZ / "data" / "dashboard"
RES = RAIZ / "data" / "resultados"
ETIQUETAS = {
    "log_base8": "Escala del corredor (log t/día)", "ratio_t": "Abastecimiento en t / promedio 8 sem.",
    "ratio_t_1": "Abastecimiento en t-1 / promedio 8 sem.", "cv8": "Coeficiente de variación (8 sem.)",
    "pend8_rel": "Tendencia relativa (8 sem.)", "activas8": "Semanas activas (de 8)",
    "activas26": "Fracción de semanas activas (26)", "n_envios8": "Cargas por semana (prom. 8)",
    "n_alimentos8": "Productos por semana (prom. 8)", "antig_sem": "Antigüedad del corredor (semanas)",
    "dist_km": "Distancia origen-central (km)", "frac_caida_mercado_t": "Fracción en caída en la central (t)",
    "frac_caida_nacional_t": "Fracción en caída en el país (t)", "festivos_t1": "Festivos en t+1",
    "sem_sin": "Semana del año (seno)", "sem_cos": "Semana del año (coseno)",
    "sh_verduras": "Participación verduras", "sh_frutas": "Participación frutas",
    "sh_tuberculos": "Participación tubérculos", "sh_granos": "Participación granos",
    "sh_lacteos": "Participación lácteos", "sh_carnes": "Participación carnes",
    "sh_pescados": "Participación pescados", "mismo_depto": "Mismo departamento", "local": "Corredor local",
    "pdet_origen": "Origen PDET", "mercado": "Central mayorista", "depto_origen": "Departamento de origen",
    "grupo_dom": "Grupo de alimentos dominante", "frac_caida_mpio_t": "Fracción en caída del municipio (t)",
    "frac_caida_vec_t": "Fracción en caída en municipios vecinos (t)",
    "tasa_int_vec26": "Tasa de interrupción de vecinos (26 sem.)",
    "tasa_int_mpio26": "Tasa de interrupción del municipio (26 sem.)", "cod_mpio": "Municipio de origen (DIVIPOLA)",
    "interrupcion_t1": "Interrupción en t+1 (objetivo)", "ratio_t1": "Abastecimiento en t+1 / promedio 8 sem.",
}


def eda(panel: pd.DataFrame):
    SAL.mkdir(parents=True, exist_ok=True)
    num = C.LOG + C.SPL + C.NUM
    cols = ["fila", "semana_t1", "anio_t1"] + C.CAT + C.ESP_TE + C.BIN + num + [C.OBJ_CLF, "ratio_t1"]
    muestra = panel.groupby("anio_t1").sample(n=10_000, random_state=C.SEMILLA)[cols].sort_values("fila")
    for c in num + ["ratio_t1"]:
        muestra[c] = muestra[c].astype("float32")
    muestra.to_parquet(SAL / "eda_muestra.parquet", index=False)

    sem = (panel.groupby(["semana_t1", "mercado", "depto_origen"], observed=True)
           .agg(corredores=(C.OBJ_CLF, "size"), interrupciones=(C.OBJ_CLF, "sum"),
                ratio_medio=("ratio_t1", "mean")).reset_index())
    sem.to_parquet(SAL / "eda_semanal.parquet", index=False)

    falt = panel[C.X_COLS + ["anio_t1"]].assign(**{c: panel[c].isna() for c in C.X_COLS}) \
        .groupby("anio_t1")[C.X_COLS].mean().T * 100
    falt.columns = falt.columns.astype(str)
    falt.reset_index(names="variable").to_parquet(SAL / "eda_faltantes.parquet", index=False)

    filas = []
    for c in num:
        v = panel[c].dropna()
        q1, q3 = v.quantile([0.25, 0.75])
        iqr = q3 - q1
        filas.append({"variable": c, "etiqueta": ETIQUETAS.get(c, c), "q1": q1, "mediana": v.median(), "q3": q3,
                      "lim_inf": q1 - 1.5 * iqr, "lim_sup": q3 + 1.5 * iqr,
                      "pct_atipicos": 100 * ((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).mean()})
    pd.DataFrame(filas).to_parquet(SAL / "eda_atipicos.parquet", index=False)

    dic = []
    for c in C.X_COLS + [C.OBJ_CLF, "ratio_t1"]:
        tipo = "categórica" if c in C.CAT + C.ESP_TE else "binaria" if c in C.BIN + [C.OBJ_CLF] else "numérica"
        fila = {"variable": c, "descripcion": ETIQUETAS.get(c, c), "tipo": tipo}
        if tipo == "numérica":
            fila["rango"] = f"{panel[c].min():.3g} a {panel[c].max():.3g}"
        elif tipo == "categórica":
            fila["rango"] = f"{panel[c].nunique()} categorías"
        else:
            fila["rango"] = "0 / 1"
        dic.append(fila)
    pd.DataFrame(dic).to_csv(SAL / "diccionario.csv", index=False)

    train, test = D.particion(panel)
    resumen = {"filas": int(len(panel)), "corredores": int(panel.groupby(["mercado", "cod_mpio"]).ngroups),
               "centrales": int(panel["mercado"].nunique()), "municipios": int(panel["cod_mpio"].nunique()),
               "semanas": int(panel["semana_t1"].nunique()), "inicio": str(panel["semana_t1"].min().date()),
               "fin": str(panel["semana_t1"].max().date()), "tasa_interrupcion": float(panel[C.OBJ_CLF].mean()),
               "filas_entrenamiento": int(len(train)), "filas_prueba": int(len(test)),
               "variables": len(C.X_COLS), "fin_entrenamiento": str(C.FIN_TRAIN.date()),
               "inicio_prueba": str(C.INICIO_TEST.date())}
    (SAL / "resumen_eda.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(resumen)




# ================================================================================ parte modelos
SLUG = {"AJ-KL (original)": "ajkl_original", "XGBoost": "xgboost", "Random Forest": "random_forest", "SVM": "svm",
        "Regresión logística": "regresion_logistica", "KNN": "knn", "Decision Tree": "decision_tree",
        "Naive Bayes": "naive_bayes", "LightGBM": "lightgbm", "CatBoost": "catboost"}


def lideres():
    """Tabla de líderes a partir de los metadata.json de los artefactos."""
    ART = RAIZ / "artifacts"
    metas = {s: json.loads((ART / s / "metadata.json").read_text()) for s in SLUG.values() if (ART / s / "metadata.json").exists()}
    # Tabla de líderes: métricas del bucle externo (media ± d.e.), prueba 2025 y tiempos
    filas = []
    for nombre, s in SLUG.items():
        if s not in metas:
            continue
        m = metas[s]
        ext, t25 = m["metricas_bucle_externo"], m["metricas_prueba_2025"]
        partes = m["combinacion"].split(" | ")
        filas.append({"Modelo": nombre, "Balanceo": partes[1], "Optimizador": partes[2],
                      "AUC externo": f"{ext['auc']['media']:.4f} ± {ext['auc']['de']:.4f}",
                      "AUC-PR externo": f"{ext['auc_pr']['media']:.4f} ± {ext['auc_pr']['de']:.4f}",
                      "Brier externo": f"{ext['brier']['media']:.4f}", "ECE externo": f"{ext['ece']['media']:.4f}",
                      "F1 externo": f"{ext['f1']['media']:.4f}", "AUC 2025": round(t25["auc"], 4),
                      "Inferencia (ms/1000 filas)": round(m["milisegundos_por_1000_predicciones"], 1),
                      "Artefacto (MB)": round(m["megabytes"], 2), "_auc": ext["auc"]["media"]})
    lid = pd.DataFrame(filas).sort_values("_auc", ascending=False).drop(columns="_auc")
    lid.to_parquet(SAL / "lideres.parquet", index=False)
    return metas


def modelos():
    """Tablas de la sección Modelos a partir de artifacts/ y data/resultados/ (scripts 12-15)."""
    sys.path.insert(0, str(RAIZ / "src"))
    from explicacion import shap_global
    ART = RAIZ / "artifacts"
    metas = lideres()
    # Predicciones 2025 de cada artefacto (reentrenado con todo el periodo)
    pa = pd.read_parquet(RES / "pred_prueba_artefactos.parquet")
    pred = pd.DataFrame({"fila": pa["fila"], "semana_t1": pa["semana_t1"], "y": pa[C.OBJ_CLF].astype("int8")})
    for nombre, s in SLUG.items():
        if s in metas and metas[s]["combinacion"] in pa:
            pred[s] = pa[metas[s]["combinacion"]].astype("float32")
    pred.to_parquet(SAL / "pred_2025.parquet", index=False)
    # SHAP global: media de |φ| en 300 corredor-semanas de 2025 (fondo de 40 filas)
    ref = pd.read_parquet(ART / "muestra_referencia.parquet")
    X = ref[C.X_COLS].sample(300, random_state=C.SEMILLA)
    fondo = ref[C.X_COLS].sample(40, random_state=7)
    filas = []
    for s in metas:
        ruta = SAL / f"shap_{s}.parquet"
        art = ART / s / "modelo.joblib"
        if ruta.exists() and ruta.stat().st_mtime > art.stat().st_mtime:      # el artefacto no cambió
            phi = pd.read_parquet(ruta)
        else:
            mod = joblib.load(art)
            phi = shap_global(lambda d: mod.predict_proba(d)[:, 1], X, fondo, M=8 if s == "knn" else 16)
            phi.to_parquet(ruta)
        for v, imp in phi.abs().mean().items():
            filas.append({"modelo": s, "variable": v, "importancia": float(imp)})
        print("SHAP", s, flush=True)
    pd.DataFrame(filas).to_parquet(SAL / "shap_global.parquet", index=False)


def resultados_modelos():
    """Textos y tablas de las secciones «Comparación estadística» y «Modelo original» del tablero."""
    V = RES / "validacion"
    corto = lambda c: c.split(" | ")[0]
    fmt = lambda x, d=4: f"{(round(x, d) or 0.0):.{d}f}".replace(".", ",")       # sin «-0,0000»
    res = json.loads((V / "resumen.json").read_text())
    tabla = pd.read_parquet(V / "tabla_pruebas.parquet")
    W = pd.read_parquet(V / "wilcoxon_mensual.parquet")
    t = tabla.copy()
    t["Conclusión"] = t["rechaza_h0"].map({True: "Rechaza H0", False: "No rechaza H0"})
    t = t.rename(columns={"prueba": "Prueba", "comparacion": "Comparación", "h0": "Hipótesis nula", "supuestos": "Supuestos",
                          "estadistico": "Estadístico", "p_crudo": "p crudo", "p_ajustado": "p ajustado", "efecto": "Efecto"})
    t["Estadístico"] = t["Estadístico"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    t["p crudo"] = t["p crudo"].map(lambda v: f"{v:.2e}")
    t["p ajustado"] = t["p ajustado"].map(lambda v: f"{v:.2e}")
    t[["Prueba", "Comparación", "Hipótesis nula", "Supuestos", "Estadístico", "p crudo", "p ajustado", "Efecto", "Conclusión"]].to_parquet(
        SAL / "pruebas.parquet", index=False)
    xgb = tabla[tabla["comparacion"].str.contains("XGBoost") | tabla["comparacion"].str.contains("modelos")]
    nombre_prueba = lambda s_: s_.split(" (")[0].split(" con ")[0].replace(" HLN, log-loss semanal", "")
    rechazan = sorted({nombre_prueba(x) for x in xgb.loc[xgb["rechaza_h0"], "prueba"]})
    no_rechazan = sorted({nombre_prueba(x) for x in xgb.loc[~xgb["rechaza_h0"], "prueba"]} - set(rechazan))
    w = W[(W.metrica == "auc") & W["benchmark"].str.startswith("XGBoost")].iloc[0]
    rangos = {corto(k): v for k, v in res["friedman"]["rangos"].items()}
    ext = [joblib.load(RES / "ajkl" / f"{pl}.joblib")["resumen"] for pl in ["k1", "k2", "k3", "k4"]]
    fin = joblib.load(RES / "ajkl" / "final.joblib")
    auc_orig = float(np.mean([e["auc"] for e in ext]))
    sel = pd.read_csv(RES / "seleccion_mejores.csv")
    supera_todos = bool(auc_orig > sel["auc_media"].max())
    efecto_ok = abs(w["cliff_pareado"]) >= 0.147 or abs(w["d_cohen"]) >= 0.2
    cumple = supera_todos and len(rechazan) >= 3 and efecto_ok
    extra = ""
    if (V / "produccion_2025.parquet").exists():
        prod = pd.read_parquet(V / "produccion_2025.parquet")
        px = prod[prod["benchmark"].str.startswith("XGBoost")].iloc[0]
        extra = (f" Con los modelos de producción (622 mil filas, prueba 2025) la diferencia frente a XGBoost es ΔAUC "
                 f"{fmt(px['delta'])} [IC BCa {fmt(px['auc_ic_inf'])}; {fmt(px['auc_ic_sup'])}] (DeLong con bloques de semana, "
                 f"p Holm {'< 0,001' if px['p_delong_holm'] < 0.001 else '= ' + fmt(px['p_delong_holm'], 3)}).")
    sota_txt = ""
    if (V / "comparadores_sota.parquet").exists():
        so = pd.read_parquet(V / "comparadores_sota.parquet")
        sota_txt = " Frente a los comparadores recientes: " + "; ".join(
            f"{r.comparador} ΔAUC 2025 {fmt(r.delta_2025)} (p Holm DeLong {fmt(r.p_delong_holm, 3)}), gana {r.meses_gana} de 48 meses"
            for r in so.itertuples()) + "."
    veredicto = (
        f"{'Superior' if cumple else 'No superior'} según la definición operativa de la guía (sección 6.1). "
        f"(1) AUC externo medio del AJ-KL {fmt(auc_orig)} frente a {fmt(sel['auc_media'].max())} de XGBoost, el mejor de los siete "
        f"benchmarks. (2) {len(rechazan)} pruebas distintas rechazan la igualdad frente a XGBoost con p ajustado < 0,05 (0,10 en el MCS): "
        f"{', '.join(rechazan)}; no rechazan: {', '.join(no_rechazan) if no_rechazan else 'ninguna'}. (3) Tamaño del efecto: d de "
        f"Cohen pareada {fmt(w['d_cohen'], 2)}, dominancia pareada {fmt(w['cliff_pareado'], 2)} (gana {int(w['meses_gana'])} de "
        f"{int(w['meses'])} meses) y delta de Cliff de dos muestras {fmt(w['cliff'], 2)} ({w['mag_cliff']}).{extra}{sota_txt} "
        f"En unidades de AUC la ventaja media es {fmt(w['media_dif'])} (escala del curso: trivial).")
    descripcion = (
        "El AJ-KL **combina** las probabilidades de 9 modelos ya entrenados (los 7 del curso, LightGBM y CatBoost) con un "
        "promedio ponderado cuyos pesos dependen de dónde está el corredor: país → central mayorista → central × grupo de "
        "alimentos → corredor. Los pesos de cada nodo se estiman minimizando la log-loss de las semanas anteriores, con más peso "
        "para las recientes (olvido γ), y se **encogen hacia los de su nodo padre** con una penalización KL: un corredor con pocas "
        "semanas usa casi los pesos de su central y uno con muchas se aparta solo si los datos lo justifican.\n\n"
        "**Qué es nuevo:** la jerarquía anidada propia del problema, el encogimiento KL hacia el padre, el olvido temporal y un "
        "estimador convexo con actualización multiplicativa cerrada (descenso espejo) de costo lineal en n, frente al apilamiento "
        "global, la selección dinámica por vecinos o el apilamiento jerárquico bayesiano por MCMC.")
    filas_abl = []
    for pl in ["k1", "k2", "k3", "k4", "final"]:
        c = joblib.load(RES / "ajkl" / f"{pl}.joblib")["comparacion"].set_index("variante")
        d = c[["auc", "auc_pr", "brier", "ece"]] - c.loc["AJ-KL (completo)", ["auc", "auc_pr", "brier", "ece"]]
        filas_abl.append(d.assign(pliegue=pl))
    abl = pd.concat(filas_abl).drop(index="AJ-KL (completo)")
    tab = abl.groupby(level=0)[["auc", "auc_pr", "brier", "ece"]].mean().reset_index().rename(
        columns={"variante": "Variante o regla", "auc": "ΔAUC", "auc_pr": "ΔAUC-PR", "brier": "ΔBrier", "ece": "ΔECE"})
    tab["Pliegues en que pierde AUC frente al AJ-KL"] = tab["Variante o regla"].map((abl["auc"] < 0).groupby(level=0).sum())
    for col in ["ΔAUC", "ΔAUC-PR", "ΔBrier", "ΔECE"]:
        tab[col] = tab[col].round(5)
    tab.sort_values("ΔAUC").to_parquet(SAL / "ablacion.parquet", index=False)
    hist = pd.concat([joblib.load(RES / "ajkl" / f"{pl}.joblib")["historial"].assign(pliegue=pl)[["pliegue", "evaluacion", "puntaje"]]
                      for pl in ["k1", "k2", "k3", "k4", "final"]]).rename(columns={"puntaje": "auc_interno"})
    hist.to_parquet(SAL / "tpe_historial.parquet", index=False)
    # pesos por central del combinador del pliegue final
    panel = D.cargar_panel()
    centrales = pd.factorize(panel["mercado"].astype(str))[1]
    modelo = fin["modelo"]
    nombres_base = [c.split(" | ")[0] for c in fin["resumen"]["modelos_base"]]
    if len(modelo.pesos_) >= 1:
        pc = pd.DataFrame([{"central": centrales[k], **dict(zip(nombres_base, w))} for k, w in modelo.pesos_[0].items()])
    else:
        pc = pd.DataFrame([{"central": "Todas (apilamiento global)", **dict(zip(nombres_base, modelo.raiz_))}])
    pc.sort_values("central").to_parquet(SAL / "pesos_central.parquet", index=False)
    import subprocess
    salida = subprocess.run([sys.executable, "-m", "pytest", str(RAIZ / "tests"), "-q", "--no-header", "-p", "no:cacheprovider"],
                            capture_output=True, text=True).stdout.strip().splitlines()[-1]
    peor = tab.sort_values("ΔAUC").iloc[0]
    resumen = {
        "veredicto": veredicto, "cumple_r3": cumple, "descripcion_original": descripcion,
        "cd": {"rangos": rangos, "cd": res["friedman"]["CD"]},
        "lectura_cd": (f"El AJ-KL tiene el mejor rango medio ({fmt(rangos[[k for k in rangos if k.startswith('AJ-KL')][0]], 2)}). "
                       f"Los modelos a menos de una diferencia crítica ({fmt(res['friedman']['CD'], 2)}) no difieren de él según "
                       "Nemenyi, una prueba conservadora; las pruebas pareadas de la tabla siguiente tienen más potencia."),
        "lectura_lideres": ("Métricas del bucle externo de la validación anidada (media ± desviación estándar de 4 años); «AUC 2025» "
                            "es la prueba con los modelos de producción reentrenados con todo el periodo de entrenamiento. La fila "
                            "resaltada es el modelo original; LightGBM y CatBoost son comparadores recientes no vistos en clase."),
        "lectura_ablacion": (f"Lo que más reduce el AUC frente al AJ-KL completo es «{peor['Variante o regla']}» ({fmt(peor['ΔAUC'], 5)} "
                             "en promedio). Valores negativos de ΔAUC y positivos de ΔBrier o ΔECE indican que el AJ-KL es mejor."),
        "pruebas_unitarias": (f"Resultado de `pytest tests/`: **{salida}**. Las pruebas del AJ-KL verifican la recuperación de los "
                              "pesos de cada región en datos sintéticos, el óptimo del descenso espejo frente a SLSQP, los casos límite "
                              "de λ, el efecto del olvido, la herencia de nodos no vistos, la calibración, la reproducibilidad, las "
                              "entradas inválidas y `clone`.\n\n"
                              f"**Tamaño del efecto frente a XGBoost:** d de Cohen pareada {fmt(w['d_cohen'], 2)} ({w['mag_d']}); "
                              f"dominancia pareada {fmt(w['cliff_pareado'], 2)} ({w['mag_cliff_pareado']}); delta de Cliff de dos "
                              f"muestras {fmt(w['cliff'], 2)} ({w['mag_cliff']})."),
    }
    (SAL / "resumen_modelos.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=float))
    print(veredicto)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "modelos":
        modelos()
        resultados_modelos()
    elif len(sys.argv) > 1 and sys.argv[1] == "resultados":
        lideres()
        resultados_modelos()
    else:
        eda(D.cargar_panel())
