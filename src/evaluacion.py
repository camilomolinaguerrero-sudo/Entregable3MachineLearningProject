"""Métricas de clasificación y regresión, umbral de decisión y calibración."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)


def umbral_f1(y, s, n_candidatos: int = 200) -> float:
    """Umbral que maximiza F1 sobre predicciones fuera de pliegue (nunca sobre validación externa).

    Los candidatos son cuantiles de los puntajes, así sirve igual para probabilidades y para
    funciones de decisión (SVM).
    """
    y = np.asarray(y)
    s = np.asarray(s, float)
    cand = np.unique(np.quantile(s, np.linspace(0.01, 0.99, n_candidatos)))
    orden = np.argsort(-s)
    s_ord, y_ord = s[orden], y[orden]
    tp = np.cumsum(y_ord)
    P = y.sum()
    mejor, umbral = -1.0, 0.5
    for c in cand:
        k = np.searchsorted(-s_ord, -c, side="right")   # número de puntajes >= c
        if k == 0:
            continue
        f1 = 2 * tp[k - 1] / (k + P)
        if f1 > mejor:
            mejor, umbral = f1, float(c)
    return umbral


def ece(y, p, bins: int = 15) -> float:
    """Error de calibración esperado con bins de igual ancho: sum_b (n_b/n) |freq_b - conf_b|."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    n = len(p)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(total)


def curva_confiabilidad(y, p, bins: int = 15) -> pd.DataFrame:
    """Tabla del diagrama de confiabilidad (bins de igual ancho)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    df = pd.DataFrame({"bin": idx, "y": y, "p": p})
    return df.groupby("bin").agg(confianza=("p", "mean"), frecuencia=("y", "mean"), n=("y", "size")).reset_index()


class Platt:
    """Escalado de Platt: regresión logística univariada sobre el puntaje."""

    def fit(self, s, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(s, float).reshape(-1, 1), y)
        return self

    def predict(self, s):
        return self.lr_.predict_proba(np.asarray(s, float).reshape(-1, 1))[:, 1]


def metricas_clf(y, s, umbral: float, prob=None) -> dict:
    """Métricas de clasificación en un conjunto de evaluación.

    Parameters
    ----------
    y : etiquetas 0/1.
    s : puntaje continuo usado para AUC y para el umbral.
    umbral : umbral sobre `s` elegido con predicciones fuera de pliegue (máximo F1).
    prob : probabilidades (si el modelo no las da, las del escalado de Platt); se usan para
           Brier, ECE, log-loss y la regla fija p >= 0,5.
    """
    y = np.asarray(y)
    s = np.asarray(s, float)
    pred = (s >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"auc": roc_auc_score(y, s), "auc_pr": average_precision_score(y, s),
           "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "umbral": umbral}
    if prob is not None:
        prob = np.clip(np.asarray(prob, float), 1e-7, 1 - 1e-7)
        p05 = (prob >= 0.5).astype(int)
        out.update({"brier": brier_score_loss(y, prob), "ece": ece(y, prob), "log_loss": log_loss(y, prob),
                    "accuracy_05": accuracy_score(y, p05), "precision_05": precision_score(y, p05, zero_division=0),
                    "recall_05": recall_score(y, p05, zero_division=0), "f1_05": f1_score(y, p05, zero_division=0),
                    "prob_media": float(prob.mean())})
    return out


def metricas_reg(y, yhat, y_clf=None) -> dict:
    """RMSE, MAE, R² y sesgo; con `y_clf` añade el AUC de -ŷ para detectar interrupciones."""
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    out = {"rmse": float(np.sqrt(mean_squared_error(y, yhat))), "mae": mean_absolute_error(y, yhat),
           "r2": r2_score(y, yhat), "sesgo": float(np.mean(yhat - y))}
    if y_clf is not None:
        out["auc_interrupcion"] = roc_auc_score(y_clf, -yhat)
    return out
