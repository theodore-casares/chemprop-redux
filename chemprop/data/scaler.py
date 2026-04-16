"""NaN-safe StandardScaler (numpy). Used for global feature + target normalization."""

from __future__ import annotations

from typing import List

import numpy as np


class StandardScaler:
    def __init__(self, means: np.ndarray | None = None, stds: np.ndarray | None = None,
                 replace_nan_token: float = 0.0) -> None:
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), 0.0, self.means)
        self.stds = np.where(np.isnan(self.stds) | (self.stds == 0.0), 1.0, self.stds)
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = (X - self.means) / self.stds
        out = np.where(np.isnan(out), self.replace_nan_token, out)
        return out

    def inverse_transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = X * self.stds + self.means
        out = np.where(np.isnan(out), self.replace_nan_token, out)
        return out
