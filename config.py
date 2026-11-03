"""Typed, validated configuration. Unknown keys are errors, so typos cannot silently no-op."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Strict):
    raw_csv: Path
    processed: Path
    reports_dir: Path
    models_dir: Path


class Source(_Strict):
    url: str
    sha256: str


class QC(_Strict):
    length_mm: tuple[float, float]
    age_years: tuple[float, float]
    allometry_a: float
    allometry_b: float
    max_log_residual: float


class Split(_Strict):
    salt: str
    test_fraction: float
    temporal_train_max_year: int


class CV(_Strict):
    n_splits: int
    n_repeats: int


class Conformal(_Strict):
    alpha: float
    predicted_age_edges: list[float]
    min_bin_size: int


class Evaluation(_Strict):
    n_bootstrap: int
    max_age_clip: float
    missingness_shift_threshold: float


class Promotion(_Strict):
    reference_model: str
    min_interval_coverage: float


class FeatureSet(_Strict):
    numeric: list[str]
    categorical: list[str]


class Candidate(_Strict):
    name: str
    kind: Literal["dummy", "vb", "alk", "ridge_spline", "hgb"]
    feature_set: str
    selectable: bool = True
    params: dict[str, Any] = {}


class Config(_Strict):
    seed: int
    paths: Paths
    source: Source
    qc: QC
    split: Split
    cv: CV
    conformal: Conformal
    evaluation: Evaluation
    promotion: Promotion
    feature_sets: dict[str, FeatureSet]
    candidates: list[Candidate]

    @model_validator(mode="after")
    def _check_references(self) -> Config:
        names = [c.name for c in self.candidates]
        if len(set(names)) != len(names):
            raise ValueError("candidate names must be unique")
        if self.promotion.reference_model not in names:
            raise ValueError("promotion.reference_model must be a candidate")
        for c in self.candidates:
            if c.feature_set not in self.feature_sets:
                raise ValueError(f"{c.name}: unknown feature set {c.feature_set}")
        return self


def load_config(path: str | Path) -> Config:
    return Config.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def config_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
