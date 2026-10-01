"""Score classifiers. Metrics are computed from predictions. Nothing here is a placeholder number."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return None


def _safe_brier(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if y_true.size == 0:
        return None
    score = np.clip(y_score, 0.0, 1.0)
    return float(brier_score_loss(y_true, score))


def operating_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(int)
    labels = [0, 1]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    auc = _safe_auc(y_true, y_score)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "average_precision": _safe_ap(y_true, y_score),
        "brier": _safe_brier(y_true, y_score),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc_auc_defined": auc is not None,
    }


def reliability_curve(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 8) -> list[dict]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.clip(np.asarray(y_score, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "count": 0,
                    "mean_predicted": None,
                    "fraction_positive": None,
                }
            )
            continue
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": count,
                "mean_predicted": float(y_score[mask].mean()),
                "fraction_positive": float(y_true[mask].mean()),
            }
        )
    return rows


def support(y_true: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    positives = int(y_true.sum())
    total = int(y_true.size)
    return {
        "n": total,
        "positives": positives,
        "negatives": total - positives,
        "positive_rate": float(positives / total) if total else None,
    }
