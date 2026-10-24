"""Sección 3: Modelos (líderes, predicción en vivo con SHAP local, diagnóstico, comparación estadística
y modelo original)."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update

import data as DT
import plots as PL
from explicacion import shap_permutaciones

dash.register_page(__name__, path="/modelos", name="Modelos", title="Modelos · Corredores de abastecimiento", order=2)

ETQ = dict(zip(DT.diccionario()["variable"], DT.diccionario()["descripcion"])) if len(DT.diccionario()) else {}
_DISP = DT.modelos_disponibles()
ESQUEMA = DT.metadata(_DISP[0])["esquema_entrada"] if _DISP else []
COLS = [c["nombre"] for c in ESQUEMA]
ID = lambda c: f"campo-{c}"
PSEUDOCODIGO = """Entrada: probabilidades fuera de muestra P (n × M) de los 9 modelos base, etiquetas y,
         semanas t, códigos de la jerarquía (central, central × grupo, corredor),
         hiperparámetros λ, γ, profundidad; tolerancia 1e-8, ε; calibrar
Salida:  pesos w_r por nodo de la jerarquía y calibrador beta (opcional)

1. P ← recortar(P, ε, 1 − ε);  a_i ← γ^(t_max − t_i)                 olvido temporal
2. w_raíz ← DescensoEspejo(P, y, a, padre = uniforme, μ = λ / Σ a_i)
3. para cada nivel l = 1 .. profundidad − 1 y cada nodo r del nivel:
       w_r ← DescensoEspejo(P_r, y_r, a_r, padre = w_π(r), μ = λ / Σ_{i∈r} a_i)
4. si calibrar: (a, b, c) ← Newton ponderado de y sobre [log q, −log(1 − q)], q_i = w_nodo(i)ᵀ p_i

DescensoEspejo(P, y, a, padre, μ):              minimiza F(w) = f(w) + μ KL(w ‖ padre) en el símplex
   w ← padre;  η ← 1
   repetir:
       g ← (1/Σa) Σ_i a_i (q_i − y_i) / (q_i (1 − q_i)) p_i          gradiente de la log-loss
       si brecha de Frank-Wolfe de F en w < tolerancia: parar      (F(w) − F* ≤ brecha)
       repetir:                                                     retroceso
           log w' ← (log w − η g + η μ log padre) / (1 + η μ) − c   paso cerrado (KKT)
           si F(w') ≤ F(w): salir;  si no: η ← η / 2
       si se estanca: paso de Frank-Wolfe hacia el vértice argmin ∇F con búsqueda lineal
       w ← w';  η ← 1,5 η
   devolver w

Predicción: nodo más profundo con pesos (o su ancestro); q ← w_rᵀ p(x); p̂ ← calibrador(q)"""


def tarjeta(titulo, cuerpo, clase=""):
    return dbc.Card([dbc.CardHeader(titulo), dbc.CardBody(cuerpo)], className=clase)


def selector_modelo(id_, valor="ajkl_original"):
    disp = DT.modelos_disponibles()
    return dcc.Dropdown(id=id_, options=[{"label": DT.NOMBRES[m], "value": m} for m in disp],
                        value=valor if valor in disp else (disp[0] if disp else None), clearable=False)


def campo(col: dict):
    nombre = col["nombre"]
    etiqueta = ETQ.get(nombre, nombre)
    ident = ID(nombre)
    if col["tipo"] == "categórica":
        ctrl = dcc.Dropdown(id=ident, options=col["categorias"][:400], value=col["categorias"][0], clearable=False)
    elif col["tipo"] == "binaria":
        ctrl = dbc.RadioItems(id=ident, options=[{"label": "No", "value": 0}, {"label": "Sí", "value": 1}], value=0, inline=True)
    else:
        ctrl = dbc.Input(id=ident, type="number", value=round(col["mediana"], 4), step="any")
    ayuda = (f"rango de entrenamiento {col['min']:.3g} a {col['max']:.3g}" if col["tipo"] == "numérica" else "")
    return dbc.Col([dbc.Label(etiqueta, html_for=ident, className="mb-0"), ctrl,
                    html.Small(ayuda, className="text-secondary")], xs=12, sm=6, lg=4, xl=3, className="mb-2")


def layout(**_):
    disp = DT.modelos_disponibles()
    if not disp:
        return dbc.Alert("Faltan los artefactos de los modelos (ejecute scripts/13_artefactos.py).", color="warning")
    lideres = DT.tabla("lideres")
    esquema = DT.metadata(disp[0])["esquema_entrada"]
    R = DT.resultados()
    return html.Div([
        html.H2("Modelos", className="mt-3"),
        dbc.Tabs([
            dbc.Tab(label="Tabla de líderes", tab_id="t-lideres", children=[
                tarjeta("Mejor combinación de cada modelo y modelo original (bucle externo de la validación anidada)", [
                    dash_table.DataTable(
                        id="tabla-lideres", data=lideres.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in lideres.columns],
                        sort_action="native", filter_action="native", page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": 12, "padding": "5px", "textAlign": "left", "fontFamily": "Inter, Segoe UI, sans-serif"},
                        style_header={"backgroundColor": "#eef3f8", "fontWeight": 600},
                        style_data_conditional=[{"if": {"filter_query": '{Modelo} contains "AJ-KL"'},
                                                 "backgroundColor": "#fff4e6", "fontWeight": 600}]),
                    html.Div(R.get("lectura_lideres", ""), className="lectura mt-2"),
                ]),
            ]),
            dbc.Tab(label="Predicción en vivo", tab_id="t-pred", children=[
                dbc.Row([
                    dbc.Col(tarjeta("Datos del corredor-semana", [
                        dbc.Row([
                            dbc.Col([dbc.Label("Modelo", html_for="m-pred"), selector_modelo("m-pred")], md=5),
                            dbc.Col(dbc.Button([html.I(className="bi bi-shuffle me-1"), "Cargar un corredor real de 2025"],
                                               id="b-ejemplo", color="secondary", outline=True, className="mt-4"), md=4),
                            dbc.Col(dbc.Button([html.I(className="bi bi-lightning me-1"), "Predecir"], id="b-predecir",
                                               color="primary", className="mt-4 w-100"), md=3),
                        ], className="mb-2"),
                        html.Div(id="txt-ejemplo", className="text-secondary small mb-2"),
                        dbc.Row([campo(c) for c in esquema]),
                    ]), lg=7),
                    dbc.Col([
                        dbc.Alert(id="pred-error", color="danger", is_open=False),
                        dcc.Loading(html.Div(id="pred-salida")),
                        dcc.Loading(dcc.Graph(id="g-shap-local", figure=PL.vacia("Pulse «Predecir» para ver la explicación local"))),
                        dcc.Loading(dcc.Graph(id="g-aportes", figure=PL.vacia("Con el AJ-KL se muestra el aporte de cada modelo base"))),
                    ], lg=5),
                ]),
            ]),
            dbc.Tab(label="Diagnóstico", tab_id="t-diag", children=[
                tarjeta("Modelo", [selector_modelo("m-diag"), html.Small("Se compara siempre con XGBoost, el mejor benchmark.",
                                                                        className="text-secondary")]),
                dbc.Row([dbc.Col(tarjeta("Curvas ROC y precisión-recall", [dcc.Loading(dcc.Graph(id="g-roc"))]), lg=7),
                         dbc.Col(tarjeta("Matriz de confusión", [dcc.Graph(id="g-conf")]), lg=5)]),
                dbc.Row([dbc.Col(tarjeta("Calibración", [dcc.Graph(id="g-cal"), html.Div(id="l-cal", className="lectura")]), lg=6),
                         dbc.Col(tarjeta("Interpretabilidad global (SHAP)", [dcc.Graph(id="g-shap-global"),
                                                                             html.Div(id="l-shap", className="lectura")]), lg=6)]),
            ]),
            dbc.Tab(label="Comparación estadística", tab_id="t-comp", children=[
                tarjeta("Diagrama de diferencia crítica (Friedman-Nemenyi, 48 meses del bucle externo, modelo original incluido)", [
                    dcc.Graph(figure=PL.diagrama_cd(pd.Series(R["cd"]["rangos"]), R["cd"]["cd"],
                                                    "Rango medio del AUC mensual (1 = mejor)") if R.get("cd") else PL.vacia()),
                    html.Div(R.get("lectura_cd", ""), className="lectura"),
                ]),
                tarjeta("Pruebas de hipótesis", [tabla_pruebas()]),
            ]),
            dbc.Tab(label="Modelo original (AJ-KL)", tab_id="t-orig", children=seccion_original(R)),
        ], active_tab="t-lideres", className="mb-3"),
    ])


def tabla_pruebas():
    t = DT.tabla("pruebas")
    if t.empty:
        return html.Div("Pendiente.")
    return dash_table.DataTable(data=t.to_dict("records"), columns=[{"name": c, "id": c} for c in t.columns],
                                sort_action="native", filter_action="native", page_size=12,
                                style_table={"overflowX": "auto"}, style_cell={"fontSize": 12, "padding": "5px",
                                                                               "textAlign": "left", "whiteSpace": "normal",
                                                                               "height": "auto"},
                                style_header={"backgroundColor": "#eef3f8", "fontWeight": 600})


def seccion_original(R):
    hist = DT.tabla("tpe_historial")
    abl = DT.tabla("ablacion")
    afines = DT.tabla("metodos_afines")
    return [
        html.Div([html.Strong("Veredicto: "), R.get("veredicto", "pendiente")], className="veredicto mb-3"),
        dbc.Row([
            dbc.Col(tarjeta("Idea", [dcc.Markdown(R.get("descripcion_original", ""))]), lg=6),
            dbc.Col(tarjeta("Pseudocódigo", [html.Div(html.Pre(PSEUDOCODIGO), className="pseudocodigo")]), lg=6),
        ]),
        tarjeta("Diferencias con la literatura (métodos afines por familia algorítmica)", [
            dash_table.DataTable(data=afines.to_dict("records"), columns=[{"name": c, "id": c} for c in afines.columns],
                                 style_cell={"fontSize": 12, "textAlign": "left", "whiteSpace": "normal", "height": "auto",
                                             "padding": "5px"},
                                 style_header={"backgroundColor": "#eef3f8", "fontWeight": 600}) if len(afines) else html.Div()]),
        tarjeta("Pesos del combinador por central mayorista (pliegue final)", [
            dcc.Graph(figure=PL.mapa_pesos(DT.tabla("pesos_central")) if len(DT.tabla("pesos_central")) else PL.vacia()),
            html.Div("Cada fila es una central y cada columna un modelo base; el color es el peso del modelo en el pool de esa "
                     "central. Los pesos cambian entre centrales: esa heterogeneidad es la que aprovecha la jerarquía.",
                     className="lectura original")]),
        dbc.Row([
            dbc.Col(tarjeta("Convergencia de la optimización (TPE, 24 evaluaciones)", [
                dcc.Graph(figure=PL.convergencia_tpe(hist) if len(hist) else PL.vacia())]), lg=6),
            dbc.Col(tarjeta("Ablación: aporte de cada componente (ΔAUC frente al modelo completo)", [
                dash_table.DataTable(data=abl.to_dict("records"), columns=[{"name": c, "id": c} for c in abl.columns],
                                     sort_action="native", style_cell={"fontSize": 12, "padding": "4px", "textAlign": "left"},
                                     style_header={"backgroundColor": "#eef3f8", "fontWeight": 600}) if len(abl) else html.Div("Pendiente."),
                html.Div(R.get("lectura_ablacion", ""), className="lectura original mt-2")]), lg=6),
        ]),
        tarjeta("Pruebas unitarias y tamaño del efecto", [dcc.Markdown(R.get("pruebas_unitarias", ""))]),
    ]


# ------------------------------------------------------------------------------- callbacks
@callback(*[Output(ID(c), "value") for c in COLS], Output("txt-ejemplo", "children"),
          Input("b-ejemplo", "n_clicks"), prevent_initial_call=True)
def cargar_ejemplo(n):
    ej = DT.ejemplos()
    if ej.empty:
        return *[no_update] * len(COLS), "No hay ejemplos disponibles."
    fila = ej.sample(1, random_state=int(n)).iloc[0]
    valores = []
    for c in COLS:
        v = fila[c]
        valores.append(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
    real = "sí" if fila["interrupcion_t1"] == 1 else "no"
    return *valores, f"Corredor {fila['cod_mpio']} → {fila['mercado']} (fila {int(fila['fila'])}). ¿Hubo interrupción en t+1? {real}."


FRACCIONES = ("sh_", "frac_", "tasa_", "activas26")


def coma(x, d):
    """Número con coma decimal (convención en español)."""
    return f"{x:.{d}f}".replace(".", ",")


def validar(nombres, valores, esquema):
    """Errores (bloquean la predicción) para valores vacíos, no numéricos o imposibles; avisos para
    valores fuera del rango de entrenamiento (la predicción extrapola)."""
    por_nombre = {c["nombre"]: c for c in esquema}
    fila, errores, avisos = {}, [], []
    for nombre, v in zip(nombres, valores):
        c = por_nombre[nombre]
        etq = ETQ.get(nombre, nombre)
        if v is None or v == "":
            errores.append(f"«{etq}» está vacío.")
            continue
        if c["tipo"] == "numérica":
            try:
                v = float(v)
            except (TypeError, ValueError):
                errores.append(f"«{etq}» debe ser numérico.")
                continue
            if not np.isfinite(v):
                errores.append(f"«{etq}» debe ser un número finito.")
            elif c["min"] >= 0 and v < 0:
                errores.append(f"«{etq}» no puede ser negativo.")
            elif nombre.startswith(FRACCIONES) and v > 1:
                errores.append(f"«{etq}» es una fracción y debe estar entre 0 y 1.")
            elif not (c["min"] <= v <= c["max"]):
                avisos.append(f"«{etq}» = {v:g} está fuera del rango de entrenamiento [{c['min']:.3g}; {c['max']:.3g}].")
        elif c["tipo"] == "categórica" and str(v) not in c["categorias"]:
            errores.append(f"«{etq}» tiene una categoría no vista en el entrenamiento.")
        elif c["tipo"] == "binaria" and v not in (0, 1):
            errores.append(f"«{etq}» debe ser 0 o 1.")
        fila[nombre] = v
    return fila, errores, avisos


@callback(Output("pred-salida", "children"), Output("g-shap-local", "figure"), Output("g-aportes", "figure"),
          Output("pred-error", "children"), Output("pred-error", "is_open"), Input("b-predecir", "n_clicks"), State("m-pred", "value"),
          *[State(ID(c), "value") for c in COLS], prevent_initial_call=True)
def predecir(n, slug, *valores):
    try:
        meta = DT.metadata(slug)
        esquema = meta["esquema_entrada"]
        fila, errores, avisos = validar(COLS, valores, esquema)
        if errores:
            return no_update, no_update, no_update, html.Ul([html.Li(e) for e in errores]), True
        orden = [c["nombre"] for c in esquema]
        x = pd.DataFrame([fila])[orden]
        fondo = DT.ejemplos()[orden].sample(40, random_state=7)
        for c in orden:
            x[c] = x[c].astype(fondo[c].dtype)
        mod = DT.modelo(slug)
        f = lambda d: mod.predict_proba(d)[:, 1]
        p = float(f(x)[0])
        umbral = float(meta["metricas_prueba_2025"]["umbral"])
        alerta = p >= umbral
        phi = shap_permutaciones(f, x, fondo, M=24 if slug == "knn" else 48)
        base = p - phi.sum()
        aviso = (dbc.Alert([html.Strong("La predicción extrapola: "), html.Ul([html.Li(a) for a in avisos], className="mb-0")],
                           color="warning", className="small") if avisos else None)
        salida = dbc.Card(dbc.CardBody([aviso,
            html.Div(DT.NOMBRES[slug], className="text-secondary small"),
            html.Div(f"{coma(100 * p, 1)} %", className="kpi-valor", style={"fontSize": "2.2rem"}),
            html.Div("probabilidad estimada de interrupción en la semana siguiente"),
            dbc.Progress(value=100 * p, color="danger" if alerta else "secondary", className="my-2",
                         label=f"{100 * p:.0f} %", style={"height": "18px"}),
            dbc.Badge("ALERTA: interrupción probable" if alerta else "Sin alerta", color="danger" if alerta else "success",
                      className="fs-6"),
            html.Div(f"Umbral de alerta {coma(umbral, 3)} (máximo F1 en la validación interna). La probabilidad media de "
                     f"referencia es {coma(100 * base, 1)} %; las barras muestran cuánto suma o resta cada variable.",
                     className="small text-secondary mt-2"),
        ]), className="mb-2")
        aportes = PL.vacia("El aporte por modelo base solo existe para el AJ-KL")
        if hasattr(mod, "pesos"):
            w = mod.pesos(x)[0]
            pb = mod.probabilidades_base(x)[0]
            aportes = PL.aportes_base(w, pb, [DT.NOMBRES.get(DT.SLUG_COMBINACION.get(o, ""), o.split(" | ")[0]) for o in mod.orden])
        return salida, PL.barras_shap(phi, ETQ, "Explicación local (Shapley por permutaciones, 12 variables con mayor efecto)"), aportes, "", False
    except Exception as e:  # noqa: BLE001 — se informa al usuario sin detener la aplicación
        return no_update, no_update, no_update, f"No se pudo calcular la predicción: {e}", True


@callback(Output("g-roc", "figure"), Output("g-conf", "figure"), Output("g-cal", "figure"), Output("l-cal", "children"),
          Output("g-shap-global", "figure"), Output("l-shap", "children"), Input("m-diag", "value"))
def diagnostico(slug):
    pred = DT.tabla("pred_2025")
    if pred.empty or slug not in pred:
        f = PL.vacia()
        return f, f, f, "", f, ""
    y = pred["y"].to_numpy()
    nombres = {slug: DT.NOMBRES[slug]}
    if slug != "xgboost" and "xgboost" in pred:
        nombres["xgboost"] = "XGBoost"
    scores = {nombres[s]: pred[s].to_numpy() for s in nombres}
    meta = DT.metadata(slug)
    m25 = meta["metricas_prueba_2025"]
    roc = PL.roc_pr(y, scores)
    conf = PL.matriz_confusion(y, pred[slug].to_numpy(), float(m25["umbral"]), DT.NOMBRES[slug])
    cal = PL.calibracion(y, scores)
    l_cal = (f"ECE de {DT.NOMBRES[slug]} en 2025: {coma(m25['ece'], 4)}; Brier {coma(m25['brier'], 4)}. Cuanto más cerca de la "
             "diagonal, más se puede leer la probabilidad como frecuencia esperada de interrupciones.")
    sg = DT.tabla("shap_global")
    sg = sg[sg["modelo"] == slug]
    if len(sg):
        imp = sg.set_index("variable")["importancia"]
        fig = PL.barras_shap(imp, ETQ, f"Importancia global de {DT.NOMBRES[slug]}: media de |SHAP| en 300 casos de 2025",
                             eje="Media de |SHAP| (puntos de probabilidad)", signo=False)
        l_shap = (f"Las variables que más mueven la probabilidad son «{ETQ.get(imp.idxmax(), imp.idxmax())}» y "
                  f"«{ETQ.get(imp.drop(imp.idxmax()).idxmax())}». La comparación entre modelos está en el capítulo 5 del informe.")
    else:
        fig, l_shap = PL.vacia(), ""
    return roc, conf, cal, l_cal, fig, l_shap
