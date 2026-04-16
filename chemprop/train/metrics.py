"""Metric functions. Each takes (targets, preds) list-of-lists and returns a float."""

from __future__ import annotations

from typing import Callable, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def _flatten(targets, preds):
    t = np.asarray(targets, dtype=float).ravel()
    p = np.asarray(preds, dtype=float).ravel()
    mask = ~np.isnan(t)
    return t[mask], p[mask]


def auc(targets, preds) -> float:
    t, p = _flatten(targets, preds)
    if len(np.unique(t)) < 2:
        return float("nan")
    return roc_auc_score(t, p)


def prc_auc(targets, preds) -> float:
    t, p = _flatten(targets, preds)
    if len(np.unique(t)) < 2:
        return float("nan")
    return average_precision_score(t, p)


def accuracy(targets, preds, threshold: float = 0.5) -> float:
    t, p = _flatten(targets, preds)
    return accuracy_score(t.astype(int), (p >= threshold).astype(int))


def rmse(targets, preds) -> float:
    t, p = _flatten(targets, preds)
    return float(np.sqrt(mean_squared_error(t, p)))


def mae(targets, preds) -> float:
    t, p = _flatten(targets, preds)
    return float(mean_absolute_error(t, p))


_METRICS = {
    "auc": auc,
    "prc-auc": prc_auc,
    "accuracy": accuracy,
    "rmse": rmse,
    "mae": mae,
}


def get_metric_func(metric: str) -> Callable:
    if metric not in _METRICS:
        raise ValueError(f"unknown metric: {metric}")
    return _METRICS[metric]
