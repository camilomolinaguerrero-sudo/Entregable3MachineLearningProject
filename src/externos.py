"""Conjuntos de datos adicionales para evaluar el modelo original fuera del problema SIPSA.

Los tres son problemas de clasificación binaria con orden temporal y estructura de grupos anidados, la
misma forma que el problema principal (un evento semanal o diario por unidad, con una jerarquía de
grupos), pero de naturaleza distinta:

* Beijing: episodios de contaminación. Y = 1 si la concentración media de PM2.5 del día siguiente en la
  estación supera 75 µg/m³ (norma diaria de grado II de China). Jerarquía: estación → estación × estación
  del año → estación × mes. 12 estaciones, 2013-2017 (Zhang et al., 2017).
* Bike: caídas de demanda de bicicletas compartidas. Y = 1 si los alquileres de la hora siguiente son menos
  de la mitad del promedio de esa hora en los 7 días anteriores (la misma regla de «interrupción» del
  SIPSA). Jerarquía: franja del día → hora → hora × día laboral. Washington D. C., 2011-2012.
* Bank: suscripción de depósitos a plazo en campañas telefónicas. Y = 1 si el cliente suscribe (se excluye
  la duración de la llamada, que solo se conoce al final). Jerarquía: ocupación → ocupación × educación →
  ocupación × educación × estado civil. Portugal, mayo de 2008 a noviembre de 2010, en orden cronológico.

Cada cargador devuelve un diccionario con el DataFrame, las listas de variables numéricas y categóricas,
el objetivo, el índice temporal (entero, creciente) y los tres niveles de la jerarquía (etiquetas).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[1] / "data" / "externos"


def beijing() -> dict:
    archivos = sorted((DIR / "beijing" / "PRSA_Data_20130301-20170228").glob("PRSA_Data_*.csv"))
    h = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
    h["fecha"] = pd.to_datetime(h[["year", "month", "day"]])
    num = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]
    d = h.groupby(["station", "fecha"]).agg({**{c: "mean" for c in num}, "RAIN": "sum",
                                              "wd": lambda s: s.mode().iloc[0] if s.notna().any() else np.nan}).reset_index()
    d = d.sort_values(["station", "fecha"]).reset_index(drop=True)
    g = d.groupby("station")
    d["pm25_t1"] = g["PM2.5"].shift(-1)
    d["pm25_lag1"] = g["PM2.5"].shift(1)
    d["pm25_media3"] = g["PM2.5"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    d["pm25_media7"] = g["PM2.5"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    d["mes"] = d["fecha"].dt.month
    d["dia_semana"] = d["fecha"].dt.dayofweek
    d["mes_sin"], d["mes_cos"] = np.sin(2 * np.pi * d["mes"] / 12), np.cos(2 * np.pi * d["mes"] / 12)
    d = d.dropna(subset=["pm25_t1"]).reset_index(drop=True)
    d["y"] = (d["pm25_t1"] > 75).astype(int)
    temporada = d["mes"].map({12: "invierno", 1: "invierno", 2: "invierno", 3: "primavera", 4: "primavera", 5: "primavera",
                              6: "verano", 7: "verano", 8: "verano", 9: "otoño", 10: "otoño", 11: "otoño"})
    numericas = num + ["RAIN", "pm25_lag1", "pm25_media3", "pm25_media7", "mes_sin", "mes_cos", "dia_semana"]
    return {"nombre": "Beijing (PM2.5 > 75 al día siguiente)", "df": d, "num": numericas, "cat": ["station", "wd"],
            "y": d["y"].to_numpy(), "tiempo": ((d["fecha"] - d["fecha"].min()).dt.days // 7).to_numpy(),
            "grupos": pd.DataFrame({"n1": d["station"], "n2": d["station"] + "|" + temporada,
                                    "n3": d["station"] + "|" + temporada + "|" + d["mes"].astype(str)})}


def bike() -> dict:
    h = pd.read_csv(DIR / "bike" / "hour.csv")
    h["momento"] = pd.to_datetime(h["dteday"]) + pd.to_timedelta(h["hr"], unit="h")
    h = h.set_index("momento").asfreq("h")
    for c in ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]:
        h[c] = h[c].ffill()
    h["cnt"] = h["cnt"].fillna(0)
    h["hr"] = h.index.hour
    # promedio de la misma hora en los 7 días anteriores (disponible al cierre de la hora t)
    misma_hora = pd.concat([h["cnt"].shift(24 * k) for k in range(1, 8)], axis=1).mean(axis=1)
    d = pd.DataFrame(index=h.index)
    d["cnt_t1"] = h["cnt"].shift(-1)
    d["ref_t1"] = misma_hora.shift(-1)
    d["cnt_t"] = h["cnt"]
    d["cnt_t_1"] = h["cnt"].shift(1)
    d["razon_t"] = h["cnt"] / misma_hora.replace(0, np.nan)
    d["media24"] = h["cnt"].rolling(24, min_periods=1).mean()
    for c in ["temp", "atemp", "hum", "windspeed", "holiday", "workingday", "weathersit", "season", "weekday"]:
        d[c] = h[c]
    d["hora_t1"] = (h["hr"] + 1) % 24
    d["hora_sin"], d["hora_cos"] = np.sin(2 * np.pi * d["hora_t1"] / 24), np.cos(2 * np.pi * d["hora_t1"] / 24)
    d = d.dropna(subset=["cnt_t1", "ref_t1"]).reset_index(names="momento")
    d = d[d["ref_t1"] >= 5].reset_index(drop=True)            # horas con demanda habitual (sin madrugadas vacías)
    d["y"] = (d["cnt_t1"] < 0.5 * d["ref_t1"]).astype(int)
    franja = pd.cut(d["hora_t1"], [-1, 5, 11, 17, 23], labels=["madrugada", "mañana", "tarde", "noche"]).astype(str)
    hora = franja + "|" + d["hora_t1"].astype(str)
    numericas = ["cnt_t", "cnt_t_1", "razon_t", "media24", "ref_t1", "temp", "atemp", "hum", "windspeed", "holiday",
                 "workingday", "hora_sin", "hora_cos"]
    d["weathersit"] = d["weathersit"].astype(int).astype(str)
    d["season"] = d["season"].astype(int).astype(str)
    d["weekday"] = d["weekday"].astype(int).astype(str)
    return {"nombre": "Bike (caída de demanda en la hora siguiente)", "df": d, "num": numericas,
            "cat": ["weathersit", "season", "weekday"], "y": d["y"].to_numpy(),
            "tiempo": ((d["momento"] - d["momento"].min()).dt.days // 7).to_numpy(),
            "grupos": pd.DataFrame({"n1": franja, "n2": hora, "n3": hora + "|" + d["workingday"].astype(int).astype(str)})}


def bank() -> dict:
    d = pd.read_csv(DIR / "bank" / "bank-additional" / "bank-additional-full.csv", sep=";")
    # el archivo está en orden cronológico; cada cambio de mes marca un nuevo periodo
    d["periodo"] = (d["month"] != d["month"].shift()).cumsum() - 1
    d["y"] = (d["y"] == "yes").astype(int)
    numericas = ["age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m",
                 "nr.employed"]
    categoricas = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
    return {"nombre": "Bank (suscripción del depósito)", "df": d, "num": numericas, "cat": categoricas,
            "y": d["y"].to_numpy(), "tiempo": d["periodo"].to_numpy(),
            "grupos": pd.DataFrame({"n1": d["job"], "n2": d["job"] + "|" + d["education"],
                                    "n3": d["job"] + "|" + d["education"] + "|" + d["marital"]})}


CARGADORES = {"beijing": beijing, "bike": bike, "bank": bank}
