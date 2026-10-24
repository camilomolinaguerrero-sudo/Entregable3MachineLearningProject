"""Búsquedas de literatura reproducibles en arXiv, OpenAlex y Semantic Scholar.

Guarda, por base y cadena, el número de resultados, la fecha de consulta y los 25 primeros
registros (título, año, DOI/identificador, fuente) en data/revision/. Las búsquedas en Web of
Science se hicieron en la interfaz web (requiere acceso institucional) y se registran aparte.
"""
import json, time, datetime as dt, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "data" / "revision"
SALIDA.mkdir(parents=True, exist_ok=True)
AGENTE = {"User-Agent": "Entregable3-ML (mailto:camilomolinaguerrero@gmail.com)"}

# Cadenas formales de la matriz de novedad: (i) familia del modelo × (ii) aplicación × (iii)
# sinónimos en inglés y español. Las cadenas C1-C8 de más abajo son exploratorias (solo familia
# algorítmica) y sirven para la tabla de métodos afines.
CADENAS_FORMALES = {
    "S1_umbrales_x_abastecimiento": {
        "openalex": '("threshold ensemble" OR "multiple thresholds" OR "distribution regression" OR "distributional regression") AND ("food supply" OR "supply disruption" OR "wholesale market" OR "food distribution")',
        "arxiv": '(all:"distribution regression" OR all:"multiple thresholds") AND (all:"food supply" OR all:"supply disruption")',
        "s2": "distribution regression multiple thresholds ensemble food supply disruption wholesale market",
    },
    "S2_latente_x_interrupcion": {
        "openalex": '("latent threshold" OR "dichotomized outcome" OR "continuous outcome" OR "latent variable") AND (interruption OR disruption OR shortage) AND ("machine learning" OR "gradient boosting" OR classifier)',
        "arxiv": '(all:"latent threshold" OR all:dichotomized) AND (all:disruption OR all:shortage) AND all:classification',
        "s2": "latent continuous outcome dichotomized threshold classifier supply disruption shortage prediction",
    },
    "S3_boosting_x_abastecimiento": {
        "openalex": '(XGBoost OR "gradient boosting" OR "random forest") AND ("food supply" OR "agri-food" OR "food market" OR "wholesale market") AND (interruption OR disruption OR shortage OR "supply drop")',
        "arxiv": '(all:XGBoost OR all:"gradient boosting") AND (all:"food supply" OR all:"agri-food") AND (all:disruption OR all:shortage)',
        "s2": "XGBoost gradient boosting food supply disruption shortage prediction wholesale market",
    },
    "S4_rearreglo_bayes_x_alerta": {
        "openalex": '("monotone rearrangement" OR "isotonic regression" OR "non-crossing") AND ("empirical Bayes" OR "hierarchical shrinkage" OR "target encoding") AND (classification OR "early warning" OR forecasting)',
        "arxiv": '(all:rearrangement OR all:isotonic) AND (all:"empirical Bayes" OR all:hierarchical) AND all:classification',
        "s2": "monotone rearrangement empirical Bayes hierarchical encoding classification early warning",
    },
    "S5_es_umbrales_x_abastecimiento": {
        "openalex": '("umbral" OR "regresión distribucional" OR "ensamble" OR "aprendizaje automático") AND ("abastecimiento de alimentos" OR "centrales mayoristas" OR "seguridad alimentaria") AND (interrupción OR desabastecimiento OR caída)',
        "arxiv": 'all:abastecimiento OR all:desabastecimiento',
        "s2": "ensamble de umbrales regresión distribucional abastecimiento de alimentos centrales mayoristas desabastecimiento",
    },
    "S6_sipsa": {
        "openalex": 'SIPSA AND ("machine learning" OR "aprendizaje automático" OR predicción OR prediction OR forecasting)',
        "arxiv": 'all:SIPSA AND (all:food OR all:supply)',
        "s2": "SIPSA abastecimiento DANE machine learning prediction",
    },
    "S7_alerta_temprana_alimentos": {
        "openalex": '("early warning" OR "alerta temprana") AND ("food supply" OR "food security" OR "seguridad alimentaria" OR "food prices") AND ("gradient boosting" OR XGBoost OR "random forest" OR "machine learning")',
        "arxiv": 'all:"early warning" AND (all:"food security" OR all:"food supply") AND all:"machine learning"',
        "s2": "early warning food security food supply machine learning gradient boosting",
    },
    "S8_combinacion_exacta": {
        "openalex": '("distribution regression" OR "multiple thresholds" OR "threshold ensemble") AND ("empirical Bayes" OR hierarchical) AND (rearrangement OR isotonic) AND (supply OR abastecimiento OR food)',
        "arxiv": 'all:"distribution regression" AND all:"empirical Bayes" AND all:rearrangement',
        "s2": "threshold-smoothed distribution regression ensemble monotone rearrangement empirical Bayes corridor food supply interruption",
    },
}

CADENAS = {
    "C1_umbrales_latentes": {
        "openalex": '("multiple thresholds" OR "threshold ensemble" OR "dichotomized outcome") AND ("binary classification" OR classifier) AND (continuous OR latent)',
        "arxiv": 'all:"multiple thresholds" AND all:classification AND all:latent',
        "s2": "ensemble of classifiers at multiple thresholds of a continuous latent outcome binary event",
    },
    "C2_regresion_distribucional": {
        "openalex": '("distribution regression" OR "distributional regression") AND (boosting OR "gradient boosting" OR "machine learning") AND (classification OR "binary")',
        "arxiv": 'all:"distribution regression" AND (all:boosting OR all:classification)',
        "s2": "distribution regression binary regressions at many thresholds gradient boosting",
    },
    "C3_reordenamiento_monotono": {
        "openalex": '("monotone rearrangement" OR "isotonic projection" OR "quantile crossing") AND ("conditional distribution" OR "distribution function")',
        "arxiv": 'all:rearrangement AND all:"distribution function" AND all:monotone',
        "s2": "monotone rearrangement conditional distribution function estimates crossing",
    },
    "C4_bayes_empirico_categoricas": {
        "openalex": '("empirical Bayes" OR "hierarchical shrinkage" OR "target encoding") AND (categorical OR "high cardinality") AND (boosting OR "random forest" OR classifier)',
        "arxiv": 'all:"target encoding" AND (all:hierarchical OR all:bayes) AND all:categorical',
        "s2": "hierarchical empirical Bayes target encoding high cardinality categorical gradient boosting",
    },
    "C5_informacion_privilegiada": {
        "openalex": '("privileged information" OR "auxiliary continuous" OR "auxiliary target") AND (classification) AND (continuous OR regression)',
        "arxiv": 'all:"privileged information" AND all:classification AND all:boosting',
        "s2": "learning using privileged information continuous auxiliary label available only at training binary classification",
    },
    "C6_aplicacion_abastecimiento_en": {
        "openalex": '("food supply" OR "wholesale market" OR "agri-food supply chain" OR "food distribution") AND (disruption OR interruption OR shortage) AND ("machine learning" OR "gradient boosting" OR XGBoost)',
        "arxiv": '(all:"food supply" OR all:"supply chain") AND all:disruption AND (all:"machine learning" OR all:boosting)',
        "s2": "machine learning prediction of food supply disruptions wholesale markets",
    },
    "C7_aplicacion_abastecimiento_es": {
        "openalex": '(abastecimiento OR "centrales mayoristas" OR SIPSA) AND (interrupción OR desabastecimiento OR caída) AND ("aprendizaje automático" OR "machine learning")',
        "arxiv": 'all:SIPSA OR (all:Colombia AND all:"food supply" AND all:"machine learning")',
        "s2": "abastecimiento de alimentos centrales mayoristas Colombia aprendizaje automático interrupción",
    },
    "C8_combinacion_exacta": {
        "openalex": '("distribution regression" OR "multiple thresholds") AND ("rearrangement" OR isotonic) AND ("empirical Bayes" OR hierarchical) AND (classification OR classifier)',
        "arxiv": 'all:"distribution regression" AND all:rearrangement AND all:classification',
        "s2": "threshold-smoothed distribution regression ensemble with monotone rearrangement and empirical Bayes group encoding for binary events",
    },
}


def get(url, intentos=4):
    for k in range(intentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENTE), timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if k == intentos - 1:
                raise
            time.sleep(3 * (k + 1))


def openalex(q):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({"filter": f"title_and_abstract.search:{q}", "per-page": 25, "sort": "relevance_score:desc"})
    d = json.loads(get(url))
    regs = [{"titulo": w.get("title"), "anio": w.get("publication_year"), "id": w.get("doi") or w.get("id"),
             "fuente": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
             "citas": w.get("cited_by_count")} for w in d["results"]]
    return d["meta"]["count"], regs


def arxiv(q):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": q, "start": 0, "max_results": 25, "sortBy": "relevance"})
    raiz = ET.fromstring(get(url))
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    total = int(raiz.find("o:totalResults", ns).text)
    regs = [{"titulo": " ".join(e.find("a:title", ns).text.split()), "anio": int(e.find("a:published", ns).text[:4]),
             "id": e.find("a:id", ns).text, "fuente": "arXiv", "citas": None} for e in raiz.findall("a:entry", ns)]
    return total, regs


def semantic(q):
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode({"query": q, "limit": 25, "fields": "title,year,externalIds,venue,citationCount"})
    d = json.loads(get(url))
    regs = [{"titulo": p.get("title"), "anio": p.get("year"), "id": (p.get("externalIds") or {}).get("DOI") or p.get("paperId"),
             "fuente": p.get("venue"), "citas": p.get("citationCount")} for p in d.get("data", [])]
    return d.get("total", len(regs)), regs


if __name__ == "__main__":
    import sys
    formales = len(sys.argv) > 1 and sys.argv[1] == "formales"
    hoy = dt.date.today().isoformat()
    registro = []
    for nombre, qs in (CADENAS_FORMALES if formales else CADENAS).items():
        for base, fn, clave in [("OpenAlex", openalex, "openalex"), ("arXiv", arxiv, "arxiv"), ("Semantic Scholar", semantic, "s2")]:
            try:
                n, regs = fn(qs[clave])
            except Exception as e:  # noqa: BLE001
                n, regs = None, [{"error": str(e)}]
            registro.append({"cadena": nombre, "base": base, "consulta": qs[clave], "fecha": hoy, "resultados": n, "registros": regs})
            print(f"{nombre:32s} {base:17s} {n}", flush=True)
            time.sleep(12 if base == "Semantic Scholar" else 1.0)
    nombre_archivo = "busquedas_formales_api.json" if formales else "busquedas_api.json"
    (SALIDA / nombre_archivo).write_text(json.dumps(registro, ensure_ascii=False, indent=1))
