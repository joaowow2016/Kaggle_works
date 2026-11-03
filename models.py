"""Baselines from fisheries science and candidate pipelines. Input is always a raw DataFrame."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    SplineTransformer,
    StandardScaler,
)

from pagrus_age.config import Candidate, FeatureSet

# Columns that must never feed a model.
# age: the target. scale_*: recorded while reading the scale, so absent for exactly the fish that
# need a prediction. condition_factor: computed from length and weight using a coefficient fitted
# on this same dataset. year: a future batch would lie outside its training range.
FORBIDDEN_FEATURES = frozenset(
    {"age", "scale_readability", "scale_border", "condition_factor", "year", "date",
     "fish_id", "landing_id", "weight_qc_flag"}
)  # fmt: skip
MONOTONIC_INCREASING = frozenset({"total_length_mm", "total_weight_g"})
DERIVED = {"month_sin": "month", "month_cos": "month"}
LENGTH = "total_length_mm"


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature step, inside the pipeline so serving cannot diverge from training."""
    out = frame.copy()
    if "month" in out:
        angle = 2 * np.pi * pd.to_numeric(out["month"], errors="coerce").astype(float) / 12
        out["month_sin"] = np.sin(angle)
        out["month_cos"] = np.cos(angle)
    for column in ("gear", "sex", "maturity_stage"):
        if column in out:
            values = out[column].astype(object)
            out[column] = values.where(values.notna(), np.nan)
    return out


def input_columns(features: FeatureSet) -> list[str]:
    """Raw columns a feature set needs, resolving derived features back to their source."""
    columns = [DERIVED.get(c, c) for c in [*features.numeric, *features.categorical]]
    return list(dict.fromkeys(columns))


def assert_no_forbidden(features: FeatureSet) -> None:
    used = set(input_columns(features)) | set(features.numeric) | set(features.categorical)
    if leaked := used & FORBIDDEN_FEATURES:
        raise ValueError(f"forbidden features requested: {sorted(leaked)}")


class AgeLengthKey(RegressorMixin, BaseEstimator):
    """Age-length key: mean observed age per length bin, the standard fisheries method."""

    def __init__(self, bin_mm: float = 20.0) -> None:
        self.bin_mm = bin_mm

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> AgeLengthKey:
        bins = np.floor(X[LENGTH].to_numpy(dtype=float) / self.bin_mm).astype(int)
        table = pd.Series(np.asarray(y, dtype=float)).groupby(bins).mean()
        self.bins_ = table.index.to_numpy()
        self.means_ = table.to_numpy()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        bins = np.floor(X[LENGTH].to_numpy(dtype=float) / self.bin_mm)
        # A length bin without training fish borrows from the nearest bin that has some.
        nearest = np.abs(bins[:, None] - self.bins_[None, :]).argmin(axis=1)
        return self.means_[nearest]


def _von_bertalanffy(age: np.ndarray, l_inf: float, k: float, t0: float) -> np.ndarray:
    return l_inf * (1 - np.exp(-k * (age - t0)))


class InverseVonBertalanffy(RegressorMixin, BaseEstimator):
    """Fit length-at-age on training fish, then invert the growth curve to read age from length."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> InverseVonBertalanffy:
        (self.l_inf_, self.k_, self.t0_), _ = curve_fit(
            _von_bertalanffy,
            np.asarray(y, dtype=float),
            X[LENGTH].to_numpy(dtype=float),
            p0=(447.0, 0.204, -1.134),  # published values, used only as a starting point
            maxfev=20000,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ratio = np.minimum(X[LENGTH].to_numpy(dtype=float) / self.l_inf_, 0.995)
        return self.t0_ - np.log(1 - ratio) / self.k_


def _hgb(features: FeatureSet, params: dict[str, Any], seed: int) -> Pipeline:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan
    )
    transformers: list[tuple[str, Any, list[str]]] = [("num", "passthrough", features.numeric)]
    if features.categorical:
        transformers.append(("cat", encoder, features.categorical))
    n_numeric, n_categorical = len(features.numeric), len(features.categorical)
    monotonic = [1 if c in MONOTONIC_INCREASING else 0 for c in features.numeric]
    model = HistGradientBoostingRegressor(
        categorical_features=list(range(n_numeric, n_numeric + n_categorical)) or None,
        monotonic_cst=monotonic + [0] * n_categorical,
        early_stopping=False,
        random_state=seed,
        **params,
    )
    return make_pipeline(
        FunctionTransformer(add_derived),
        ColumnTransformer(transformers, remainder="drop"),
        model,
    )


def _ridge_spline(features: FeatureSet, params: dict[str, Any]) -> Pipeline:
    length = make_pipeline(
        SimpleImputer(strategy="median"),
        SplineTransformer(n_knots=6, degree=3, extrapolation="linear"),
    )
    transformers: list[tuple[str, Any, list[str]]] = [("length", length, [LENGTH])]
    others = [c for c in features.numeric if c != LENGTH]
    if others:
        transformers.append(
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), others)
        )
    if features.categorical:
        categorical = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="missing"),
            OneHotEncoder(handle_unknown="ignore"),
        )
        transformers.append(("cat", categorical, features.categorical))
    return make_pipeline(
        FunctionTransformer(add_derived), ColumnTransformer(transformers), Ridge(**params)
    )


def build_model(candidate: Candidate, features: FeatureSet, seed: int) -> Any:
    assert_no_forbidden(features)
    if candidate.kind == "dummy":
        return DummyRegressor(strategy="median")
    if candidate.kind == "alk":
        return AgeLengthKey(**candidate.params)
    if candidate.kind == "vb":
        return InverseVonBertalanffy()
    if candidate.kind == "ridge_spline":
        return _ridge_spline(features, candidate.params)
    return _hgb(features, candidate.params, seed)
