"""Matriz de novedad a partir de las búsquedas en Web of Science Core Collection.

Las búsquedas se ejecutaron en la interfaz web de WoS (acceso institucional del Consorcio Colciencias)
con la búsqueda avanzada (campo TS = título, resumen y palabras clave), ordenadas por relevancia; de
cada cadena se revisaron hasta 50 registros (título, autores, fuente y resumen) con dos revisores
independientes (lente del método y lente de la aplicación). Un registro marcado «relevante» por
cualquiera de los dos se sometía a tres verificadores que intentaban refutarlo. Este script consolida
los resultados guardados en data/revision/wos/ en la matriz de novedad y la lista de trabajos cercanos.
"""
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
WOS = RAIZ / "data" / "revision" / "wos"
FECHAS = {"ronda1_resultado.json": "2026-10-23", "ronda2_resultado.json": "2026-10-24"}


def main():
    filas, cercanos = [], []
    for archivo, fecha in FECHAS.items():
        ruta = WOS / archivo
        if not ruta.exists():
            continue
        for f in json.loads(ruta.read_text())["filas"]:
            cm = sum(1 for c in f["cercanos"] if c.get("clase") == "cercano_metodo")
            ca = sum(1 for c in f["cercanos"] if c.get("clase") == "cercano_aplicacion")
            filas.append({"base": "Web of Science Core Collection", "cadena": f["id"], "proposito": f["proposito"],
                          "consulta": f["consulta"], "fecha": fecha, "resultados": f["resultados"], "revisados": f["revisados"],
                          "relevantes": f["relevantes"], "cercanos_metodo": cm, "cercanos_aplicacion": ca,
                          "observacion": (f.get("observacion") or "").split(" / ")[0]})
            for c in f["cercanos"]:
                cercanos.append({"cadena": f["id"], "titulo": c["titulo"], "autores": c.get("autores", ""),
                                 "fecha": c.get("fecha", ""), "fuente": c.get("fuente", "").replace("&amp;", "&"),
                                 "clase": c.get("clase", ""), "motivo": c.get("motivo", "")})
    m = pd.DataFrame(filas).sort_values("cadena")
    m.to_csv(RAIZ / "data" / "revision" / "matriz_novedad_wos.csv", index=False)
    c = pd.DataFrame(cercanos).drop_duplicates("titulo")
    c.to_csv(RAIZ / "data" / "revision" / "wos_cercanos.csv", index=False)
    print(m[["cadena", "resultados", "revisados", "relevantes", "cercanos_metodo", "cercanos_aplicacion"]].to_string())
    print("Total:", int(m["resultados"].sum()), "resultados;", int(m["revisados"].sum()), "revisados;",
          int(m["relevantes"].sum()), "relevantes;", len(c), "trabajos cercanos distintos")


if __name__ == "__main__":
    main()
