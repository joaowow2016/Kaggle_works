"""Batch inference and drift checks over a verified, promoted model bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pagrus_age.config import QC
from pagrus_age.data import apply_qc, read_raw, sha256_file, to_canonical, validate_raw
from pagrus_age.metrics import psi

PSI_ALERT = 0.25
MISSING_ALERT = 0.20


class ModelNotPromotedError(RuntimeError):
    """No promoted model exists; serving one would bypass the promotion gate."""


def load_bundle(models_dir: str | Path, run_id: str | None = None) -> dict[str, Any]:
    """Load a bundle after checking its hash. Joblib runs code on load: trusted files only."""
    models_dir = Path(models_dir)
    if run_id is None:
        pointer = models_dir / "production.json"
        if not pointer.exists():
            raise ModelNotPromotedError(
                f"{pointer} not found: no run passed the promotion gate. "
                "Pass an explicit run ID to inspect an unpromoted model."
            )
        run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
    bundle_path = models_dir / str(run_id) / "bundle.joblib"
    manifest = json.loads((bundle_path.parent / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(bundle_path) != manifest["bundle_sha256"]:
        raise ValueError(f"{bundle_path} does not match the hash in its manifest")
    bundle: dict[str, Any] = joblib.load(bundle_path)
    return bundle


def predict_frame(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Predict from canonical columns. Optional inputs may be absent; length is required."""
    if frame["total_length_mm"].isna().any():
        raise ValueError("total_length_mm is required for every fish")
    X = frame.reindex(columns=bundle["input_columns"]).copy()
    predicted = np.clip(bundle["model"].predict(X), 0.0, bundle["max_age_clip"])
    lower, upper = bundle["conformal"].interval(predicted)
    low, high = bundle["length_range_mm"]
    return pd.DataFrame(
        {
            "age_pred": predicted,
            "age_pred_rounded": np.rint(predicted).astype(int),
            "age_lower_90": lower,
            "age_upper_90": upper,
            "length_outside_training_range": ~frame["total_length_mm"]
            .between(low, high)
            .to_numpy(),
        },
        index=frame.index,
    )


def load_input(path: str | Path, qc: QC) -> pd.DataFrame:
    """Accept the publisher's schema or canonical names; raw files get the same QC as training."""
    frame = read_raw(path)
    if "Total_lenght" in frame.columns:
        validate_raw(frame)
        return apply_qc(to_canonical(frame), qc)[0]
    return frame


def predict_file(
    bundle: dict[str, Any],
    input_path: str | Path,
    output_path: str | Path,
    qc: QC,
    unaged_only: bool = False,
) -> pd.DataFrame:
    frame = load_input(input_path, qc)
    if unaged_only and "age" in frame:
        frame = frame[frame["age"].isna()]
    predictions = predict_frame(bundle, frame)
    columns = [c for c in ("fish_id", "landing_id", "total_length_mm", "gear", "scale_readability")
               if c in frame]  # fmt: skip
    result = pd.concat([frame[columns], predictions], axis=1).assign(model_run_id=bundle["run_id"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def drift_report(bundle: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    """Compare a new batch with the training reference. Alerts are for review, not automatic."""
    columns: dict[str, Any] = {}
    alerts: list[str] = []
    for column, reference in bundle["reference"].items():
        if column not in frame:
            alerts.append(f"{column}: absent from batch")
            continue
        current = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        edges = np.unique(np.nanquantile(reference, np.linspace(0.1, 0.9, 9)))
        score = psi(reference, current, edges)
        missing_gap = float(abs(np.isnan(current).mean() - np.isnan(reference).mean()))
        columns[column] = {"psi": score, "missing_rate_gap": missing_gap}
        if score > PSI_ALERT:
            alerts.append(f"{column}: PSI {score:.2f} > {PSI_ALERT}")
        if missing_gap > MISSING_ALERT:
            alerts.append(f"{column}: missing rate gap {missing_gap:.2f} > {MISSING_ALERT}")
    return {"n": len(frame), "columns": columns, "alerts": alerts}
