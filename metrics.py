"""Metrics, trip-level bootstrap and conformal intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    rounded = np.rint(y_pred).astype(int)
    truth = np.rint(y_true).astype(int)
    agreement = (
        cohen_kappa_score(truth, rounded, weights="quadratic")
        if len(np.union1d(truth, rounded)) > 1
        else float("nan")
    )
    return {
        "n": len(y_true),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
        "exact_rate": float((rounded == truth).mean()),
        "within_1yr_rate": float((np.abs(rounded - truth) <= 1).mean()),
        "qwk": float(agreement),
    }


def _group_totals(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, inverse = np.unique(groups, return_inverse=True)
    return np.bincount(inverse, values), np.bincount(inverse).astype(float)


def cluster_bootstrap_mean(
    values: np.ndarray, groups: np.ndarray, n_boot: int, seed: int, level: float = 0.95
) -> dict[str, float]:
    """Mean with a percentile CI, resampling landings because landings are the independent unit."""
    totals, counts = _group_totals(np.asarray(values, dtype=float), groups)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(totals), size=(n_boot, len(totals)))
    means = np.divide(totals[picks].sum(axis=1), counts[picks].sum(axis=1))
    tail = (1 - level) / 2
    return {
        "estimate": float(totals.sum()) / float(counts.sum()),
        "ci_low": float(np.quantile(means, tail)),
        "ci_high": float(np.quantile(means, 1 - tail)),
    }


@dataclass
class MondrianConformal:
    """Split-conformal half-widths per predicted-age bin; residual spread grows with age."""

    alpha: float
    edges: list[float]
    min_bin_size: int
    half_widths: list[float] | None = None

    @staticmethod
    def _quantile(residuals: np.ndarray, alpha: float) -> float:
        n = len(residuals)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(residuals, level, method="higher"))

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> MondrianConformal:
        residuals = np.abs(np.asarray(y_true, dtype=float) - y_pred)
        overall = self._quantile(residuals, self.alpha)
        bins = np.digitize(y_pred, self.edges)
        self.half_widths = []
        for index in range(len(self.edges) + 1):
            chosen = residuals[bins == index]
            enough = len(chosen) >= self.min_bin_size
            self.half_widths.append(self._quantile(chosen, self.alpha) if enough else overall)
        return self

    def interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.half_widths is None:
            raise RuntimeError("fit before calling interval")
        half = np.asarray(self.half_widths)[np.digitize(y_pred, self.edges)]
        return np.maximum(y_pred - half, 0.0), y_pred + half


def psi(reference: np.ndarray, current: np.ndarray, edges: np.ndarray) -> float:
    """Population stability index over edges fixed from the reference distribution."""
    bounds = np.concatenate(([-np.inf], edges, [np.inf]))
    expected = np.histogram(reference[~np.isnan(reference)], bins=bounds)[0].astype(float)
    actual = np.histogram(current[~np.isnan(current)], bins=bounds)[0].astype(float)
    if actual.sum() == 0:
        return float("nan")
    expected = np.clip(expected / expected.sum(), 1e-4, None)
    actual = np.clip(actual / actual.sum(), 1e-4, None)
    return float(((actual - expected) * np.log(actual / expected)).sum())


def slice_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Error by subgroup. Expects columns y_true and y_pred."""
    rows = []
    for level, part in frame.groupby(by, dropna=False, observed=True):
        error = part["y_pred"] - part["y_true"]
        rows.append(
            {
                "slice": by,
                "level": "missing" if pd.isna(level) else str(level),
                "n": len(part),
                "landings": part["landing_id"].nunique(),
                "mae": float(error.abs().mean()),
                "bias": float(error.mean()),
                "within_1yr_rate": float(
                    (np.abs(np.rint(part["y_pred"]) - part["y_true"]) <= 1).mean()
                ),
            }
        )
    return pd.DataFrame(rows)
