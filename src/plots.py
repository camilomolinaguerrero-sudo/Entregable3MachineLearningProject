"""Figuras Plotly reutilizables del dashboard (y del libro). Paleta de cinco colores, plantilla
única, títulos y ejes con unidades."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

AZUL_OSC, AZUL, NARANJA, ROJO, GRIS = "#0b3d6e", "#1c7ed6", "#e8590c", "#c92a2a", "#868e96"
PALETA = [AZUL_OSC, AZUL, NARANJA, ROJO, GRIS]
EVENTOS = [("2020-03-23", "2020-05-31", "Confinamiento"), ("2021-04-28", "2021-06-15", "Paro nacional")]

pio.templates["proyecto"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12, color="#212529"),
    paper_bgcolor="white", plot_bgcolor="white", colorway=PALETA,
    xaxis=dict(gridcolor="#eef0f2", zerolinecolor="#dee2e6", linecolor="#ced4da", automargin=True),
    yaxis=dict(gridcolor="#eef0f2", zerolinecolor="#dee2e6", linecolor="#ced4da", automargin=True),
    margin=dict(l=55, r=20, t=70, b=50), hoverlabel=dict(font_size=12),
    title=dict(x=0.01, xanchor="left", font=dict(size=15)),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=11)),
))
pio.templates.default = "proyecto"


def color_modelo(nombre: str) -> str:
    return NARANJA if nombre.startswith("AJ-KL") else AZUL if nombre == "XGBoost" else GRIS


def vacia(mensaje: str = "Sin datos para los filtros elegidos") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=14, color=GRIS))
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    return fig


# ------------------------------------------------------------------------------- EDA
def serie_tasa(sem: pd.DataFrame) -> go.Figure:
    if sem.empty:
        return vacia()
    s = sem.groupby("semana_t1")[["corredores", "interrupciones"]].sum()
    s["tasa"] = 100 * s["interrupciones"] / s["corredores"]
    s["media_movil"] = s["tasa"].rolling(8, min_periods=1).mean()
    fig = go.Figure()
    for ini, fin, nombre in EVENTOS:
        fig.add_vrect(x0=ini, x1=fin, fillcolor=GRIS, opacity=0.15, line_width=0, annotation_text=nombre,
                      annotation_position="top left", annotation_font_size=10)
    fig.add_scatter(x=s.index, y=s["tasa"], name="Tasa semanal", line=dict(color=AZUL, width=1), opacity=0.6,
                    hovertemplate="%{x|%d-%m-%Y}<br>%{y:.1f} %<extra></extra>")
    fig.add_scatter(x=s.index, y=s["media_movil"], name="Media móvil de 8 semanas", line=dict(color=AZUL_OSC, width=2.5),
                    hovertemplate="%{x|%d-%m-%Y}<br>%{y:.1f} %<extra></extra>")
    fig.update_layout(title="Porcentaje de corredores con interrupción en la semana siguiente", height=360)
    fig.update_yaxes(title="Corredores con interrupción (%)").update_xaxes(title="Semana objetivo (t+1)")
    return fig


def balance_clases(m: pd.DataFrame) -> go.Figure:
    if m.empty:
        return vacia()
    c = m["interrupcion_t1"].map({0: "Sin interrupción", 1: "Interrupción"}).value_counts()
    fig = go.Figure(go.Bar(x=c.index, y=100 * c.values / c.sum(), marker_color=[GRIS if i.startswith("Sin") else ROJO for i in c.index],
                           text=[f"{100 * v / c.sum():.1f} %<br>({v:,} filas)".replace(",", ".") for v in c.values],
                           textposition="outside", hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.update_layout(title="Balance de clases del objetivo", height=330, showlegend=False)
    fig.update_yaxes(title="Corredor-semanas (%)", range=[0, 100]).update_xaxes(title="")
    return fig


def histograma(m: pd.DataFrame, var: str, etiqueta: str, recorte: bool = True) -> go.Figure:
    if m.empty or var not in m:
        return vacia()
    d = m[[var, "interrupcion_t1"]].dropna()
    if recorte and d[var].dtype.kind == "f":
        lo, hi = d[var].quantile([0.01, 0.99])
        d = d[d[var].between(lo, hi)]
    d["Clase"] = d["interrupcion_t1"].map({0: "Sin interrupción", 1: "Interrupción"})
    fig = px.histogram(d, x=var, color="Clase", barmode="overlay", histnorm="percent", nbins=50, opacity=0.65,
                       color_discrete_map={"Sin interrupción": GRIS, "Interrupción": ROJO})
    fig.update_layout(title=f"Distribución de «{etiqueta}» por clase (1 % extremo recortado)", height=340)
    fig.update_xaxes(title=etiqueta).update_yaxes(title="Corredor-semanas de la clase (%)")
    return fig


def bivariado(m: pd.DataFrame, x: str, y: str, color: str, etiquetas: dict) -> go.Figure:
    if m.empty:
        return vacia()
    d = m.sample(min(len(m), 6000), random_state=42).copy()
    if color == "interrupcion_t1":
        d[color] = d[color].map({0: "Sin interrupción", 1: "Interrupción"})
    es_cat_x = d[x].dtype == object or str(d[x].dtype) == "category"
    if es_cat_x:
        top = d[x].value_counts().index[:12]
        d = d[d[x].isin(top)]
        fig = px.box(d, x=x, y=y, color=color if d[color].nunique() <= 6 else None, points=False,
                     color_discrete_sequence=[GRIS, ROJO, AZUL, NARANJA, AZUL_OSC])
    else:
        fig = px.scatter(d, x=x, y=y, color=color, opacity=0.45,
                         color_discrete_sequence=[GRIS, ROJO, AZUL, NARANJA, AZUL_OSC],
                         color_continuous_scale="Blues")
    fig.update_layout(title=f"«{etiquetas.get(y, y)}» frente a «{etiquetas.get(x, x)}» (6 000 puntos al azar)", height=420)
    fig.update_xaxes(title=etiquetas.get(x, x)).update_yaxes(title=etiquetas.get(y, y))
    return fig


def correlacion(m: pd.DataFrame, variables: list[str], etiquetas: dict) -> go.Figure:
    if m.empty:
        return vacia()
    c = m[variables].corr(method="spearman")
    nombres = [etiquetas.get(v, v) for v in variables]
    fig = go.Figure(go.Heatmap(z=c.values, x=nombres, y=nombres, zmin=-1, zmax=1, colorscale="RdBu_r",
                               colorbar=dict(title="ρ"), hovertemplate="%{y}<br>%{x}<br>ρ = %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Correlación de Spearman entre variables numéricas", height=620)
    fig.update_xaxes(tickangle=45, tickfont=dict(size=9)).update_yaxes(tickfont=dict(size=9), autorange="reversed")
    return fig


def mapa_faltantes(falt: pd.DataFrame, etiquetas: dict) -> go.Figure:
    if falt.empty:
        return vacia()
    f = falt.set_index("variable")
    f = f[(f > 0).any(axis=1)]
    if f.empty:
        return vacia("Ninguna variable del modelo tiene valores faltantes")
    fig = go.Figure(go.Heatmap(z=f.values, x=f.columns, y=[etiquetas.get(v, v) for v in f.index], colorscale="Oranges",
                               colorbar=dict(title="%"), hovertemplate="%{y}<br>%{x}: %{z:.2f} % faltante<extra></extra>"))
    fig.update_layout(title="Valores faltantes por variable y año (% de filas)", height=360)
    fig.update_xaxes(title="Año de la semana objetivo")
    return fig


def atipicos(m: pd.DataFrame, var: str, etiqueta: str) -> go.Figure:
    if m.empty:
        return vacia()
    d = m[[var, "anio_t1"]].dropna()
    fig = px.box(d, x="anio_t1", y=var, points="outliers", color_discrete_sequence=[AZUL])
    fig.update_layout(title=f"Atípicos de «{etiqueta}» por año (regla de Tukey: 1,5 × RIC)", height=360)
    fig.update_xaxes(title="Año de la semana objetivo").update_yaxes(title=etiqueta)
    return fig


def descomposicion(serie: pd.Series) -> go.Figure:
    from statsmodels.tsa.seasonal import STL
    if len(serie) < 110:
        return vacia("La serie filtrada es demasiado corta para la descomposición (se requieren 2 años)")
    r = STL(serie, period=52, robust=True).fit()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=("Serie observada (%)", "Tendencia", "Estacionalidad anual", "Residuo"))
    for i, (comp, col) in enumerate([(serie, AZUL_OSC), (r.trend, AZUL), (r.seasonal, NARANJA), (r.resid, GRIS)], 1):
        fig.add_scatter(x=serie.index, y=comp, line=dict(color=col, width=1.5), showlegend=False, row=i, col=1)
    fig.update_layout(title="Descomposición STL de la tasa semanal de interrupción (periodo de 52 semanas)", height=620)
    return fig


def acf_pacf(serie: pd.Series, rezagos: int = 60) -> go.Figure:
    from statsmodels.tsa.stattools import acf, pacf
    if len(serie) < rezagos + 10:
        return vacia("Serie demasiado corta para la autocorrelación")
    a, p = acf(serie, nlags=rezagos), pacf(serie, nlags=rezagos)
    banda = 1.96 / np.sqrt(len(serie))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Autocorrelación (ACF)", "Autocorrelación parcial (PACF)"))
    for j, v in enumerate([a, p], 1):
        fig.add_bar(x=list(range(len(v))), y=v, marker_color=AZUL, showlegend=False, row=1, col=j)
        fig.add_hline(y=banda, line_dash="dash", line_color=ROJO, row=1, col=j)
        fig.add_hline(y=-banda, line_dash="dash", line_color=ROJO, row=1, col=j)
    fig.update_layout(title="ACF y PACF de la tasa semanal (bandas al 95 %)", height=340)
    fig.update_xaxes(title="Rezago (semanas)")
    return fig


# ------------------------------------------------------------------------------- modelos
def roc_pr(y, scores: dict[str, np.ndarray]) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Curva ROC", "Curva precisión-recall"), horizontal_spacing=0.12)
    for nombre, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        paso = max(1, len(fpr) // 2000)
        col = color_modelo(nombre)
        fig.add_scatter(x=fpr[::paso], y=tpr[::paso], name=f"{nombre} (AUC {roc_auc_score(y, s):.4f})",
                        line=dict(color=col, width=2.5 if col != GRIS else 1.5), row=1, col=1)
        paso = max(1, len(rec) // 2000)
        fig.add_scatter(x=rec[::paso], y=prec[::paso], name=nombre, showlegend=False,
                        line=dict(color=col, width=2.5 if col != GRIS else 1.5), row=1, col=2)
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=GRIS, dash="dot"), showlegend=False, row=1, col=1)
    fig.add_hline(y=float(np.mean(y)), line_dash="dot", line_color=GRIS, row=1, col=2)
    fig.update_xaxes(title="Tasa de falsos positivos", row=1, col=1).update_yaxes(title="Tasa de verdaderos positivos", row=1, col=1)
    fig.update_xaxes(title="Recall", row=1, col=2).update_yaxes(title="Precisión", row=1, col=2)
    fig.update_layout(title="Discriminación en la prueba 2025 (103 466 corredor-semanas)", height=460,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), margin=dict(t=80, b=110))
    return fig


def calibracion(y, probs: dict[str, np.ndarray], bins: int = 15) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=GRIS, dash="dot"), name="Calibración perfecta")
    for nombre, p in probs.items():
        d = pd.DataFrame({"y": y, "p": p})
        d["bin"] = pd.cut(d["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
        g = d.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size")).query("n >= 30")
        fig.add_scatter(x=g["p"], y=g["y"], mode="lines+markers", name=nombre, line=dict(color=color_modelo(nombre)),
                        customdata=g["n"], hovertemplate="predicha %{x:.2f}<br>observada %{y:.2f}<br>n = %{customdata}<extra></extra>")
    fig.update_layout(title="Diagrama de calibración en 2025 (15 intervalos de probabilidad)", height=420)
    fig.update_xaxes(title="Probabilidad predicha", range=[0, 1]).update_yaxes(title="Frecuencia observada de interrupción", range=[0, 1])
    return fig


def matriz_confusion(y, p, umbral: float, nombre: str) -> go.Figure:
    cm = confusion_matrix(y, (p >= umbral).astype(int), labels=[0, 1])
    pct = 100 * cm / cm.sum()
    texto = [[f"{cm[i, j]:,}<br>({pct[i, j]:.1f} %)".replace(",", ".") for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=cm, x=["Predice: no", "Predice: sí"], y=["Real: no", "Real: sí"], text=texto,
                               texttemplate="%{text}", colorscale="Blues", showscale=False,
                               hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>"))
    fig.update_layout(title=f"{nombre}: interrupción en t+1<br><sup>umbral {umbral:.3f} (máximo F1 interno)</sup>", height=380,
                      margin=dict(l=70, r=10, t=80, b=40))
    fig.update_yaxes(autorange="reversed")
    return fig


def barras_shap(phi: pd.Series, etiquetas: dict, titulo: str, n: int = 12,
                eje: str = "Contribución a la probabilidad de interrupción (puntos de probabilidad)", signo: bool = True) -> go.Figure:
    """Barras horizontales de valores de Shapley. Con `signo`, rojo sube la probabilidad y azul la baja."""
    s = phi.reindex(phi.abs().sort_values(ascending=False).index[:n])[::-1]
    colores = [ROJO if v > 0 else AZUL for v in s.values] if signo else [AZUL_OSC] * len(s)
    fig = go.Figure(go.Bar(x=s.values, y=[etiquetas.get(i, i) for i in s.index], orientation="h", marker_color=colores,
                           hovertemplate="%{y}: %{x:+.4f}<extra></extra>" if signo else "%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(title=titulo, height=440, showlegend=False, margin=dict(l=10, r=20, t=60, b=50))
    fig.update_xaxes(title=eje)
    return fig


def diagrama_cd(rangos: pd.Series, cd: float, titulo: str) -> go.Figure:
    """Diagrama de diferencia crítica (Demšar, 2006): rango medio de cada modelo y la CD de Nemenyi."""
    r = rangos.sort_values()
    fig = go.Figure()
    y = np.arange(len(r))[::-1]
    fig.add_scatter(x=r.values, y=y, mode="markers+text", text=[f"{n} ({v:.2f})" for n, v in r.items()],
                    textposition="middle right", marker=dict(size=12, color=[color_modelo(n) for n in r.index]),
                    showlegend=False, hovertemplate="%{text}<extra></extra>")
    mejor = r.iloc[0]
    fig.add_shape(type="rect", x0=mejor, x1=mejor + cd, y0=-0.6, y1=len(r) - 0.4, fillcolor=NARANJA, opacity=0.08, line_width=0)
    fig.add_annotation(x=mejor + cd, y=len(r) - 0.4, text=f"CD = {cd:.2f}", showarrow=False, yanchor="bottom", font=dict(color=NARANJA))
    fig.update_layout(title=titulo, height=380)
    fig.update_xaxes(title="Rango medio (1 = mejor)", range=[0.5, len(r) + 2.5]).update_yaxes(visible=False)
    return fig


def convergencia_tpe(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for pl, g in hist.groupby("pliegue"):
        g = g.sort_values("evaluacion")
        fig.add_scatter(x=g["evaluacion"], y=g["auc_interno"].cummax(), mode="lines+markers", name=pl,
                        line=dict(color=NARANJA if pl == "final" else GRIS, width=2.5 if pl == "final" else 1.2))
    fig.update_layout(title="Convergencia de TPE: mejor AUC interno acumulado del modelo original por pliegue externo", height=360)
    fig.update_xaxes(title="Evaluación (presupuesto de 24)").update_yaxes(title="Mejor AUC interno")
    return fig


def aportes_base(pesos, probs, nombres) -> go.Figure:
    """Descomposición exacta del pool del AJ-KL: q = Σ_m w_m p_m (antes de la calibración)."""
    orden = np.argsort(pesos * probs)
    fig = go.Figure(go.Bar(x=(pesos * probs)[orden], y=[nombres[i] for i in orden], orientation="h", marker_color=NARANJA,
                           customdata=np.c_[pesos[orden], probs[orden]],
                           hovertemplate="%{y}<br>peso %{customdata[0]:.3f} × probabilidad %{customdata[1]:.3f} = %{x:.4f}<extra></extra>"))
    fig.update_layout(title=f"Aporte de cada modelo base al pool del AJ-KL (suma = {np.sum(pesos * probs):.3f})", height=380,
                      showlegend=False, margin=dict(l=10, r=20, t=60, b=50))
    fig.update_xaxes(title="Peso del nodo × probabilidad del modelo base")
    return fig


def mapa_pesos(t: pd.DataFrame) -> go.Figure:
    """Mapa de calor de los pesos del combinador por central (filas) y modelo base (columnas)."""
    m = t.set_index("central")
    fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=m.index, colorscale="Oranges", zmin=0,
                               colorbar=dict(title="peso"), hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>"))
    fig.update_layout(title="Pesos del pool por central mayorista", height=max(420, 18 * len(m)))
    fig.update_xaxes(tickangle=30).update_yaxes(tickfont=dict(size=10))
    return fig
