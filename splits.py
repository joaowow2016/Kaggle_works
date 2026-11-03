"""Group-aware splits. Fish from one landing are correlated, so a landing never spans a split."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def _stable_rank(landing_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{landing_id}".encode()).hexdigest()


def held_out_landings(df: pd.DataFrame, salt: str, fraction: float) -> set[str]:
    """Choose test landings per gear by hash: independent of row order, age and any model."""
    landings = df[["landing_id", "gear"]].drop_duplicates("landing_id")
    selected: set[str] = set()
    for _, group in landings.groupby("gear"):
        ordered = sorted(group["landing_id"], key=lambda value: _stable_rank(value, salt))
        selected.update(ordered[: max(1, round(len(ordered) * fraction))])
    return selected


def repeated_group_folds(
    groups: np.ndarray, n_splits: int, n_repeats: int, seed: int
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    placeholder = np.zeros(len(groups))
    for repeat in range(n_repeats):
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold, (train, valid) in enumerate(splitter.split(placeholder, groups=groups)):
            yield repeat, fold, train, valid
