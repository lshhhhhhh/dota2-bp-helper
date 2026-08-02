from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    return {
        "auc": roc_auc(y, p),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "ece_10": expected_calibration_error(y, p, bins=10),
    }


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(y_score, dtype=np.float64)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return error

