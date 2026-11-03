"""Training run: select on development landings, then evaluate once on held-out landings."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from pagrus_age.config import Candidate, Config, config_hash, load_config
from pagrus_age.data import ingest, sha256_file
from pagrus_age.metrics import (
    MondrianConformal,
    cluster_bootstrap_mean,
    regression_metrics,
    slice_table,
)
from pagrus_age.models import build_model, input_columns
from pagrus_age.reporting import plot_calibration, plot_model_comparison, write_model_card
from pagrus_age.splits import held_out_landings, repeated_group_folds

DRIFT_COLUMNS = ["total_length_mm", "depth_min_m", "depth_max_m", "month"]
AGE_BINS = [-np.inf, 2, 5, 8, 12, np.inf]
AGE_LABELS = ["1-2", "3-5", "6-8", "9-12", "13+"]


@dataclass
class CandidateCV:
    candidate: Candidate
    fold_mae: list[float]
    oof_first_repeat: np.ndarray

    @property
    def mean_mae(self) -> float:
        return float(np.mean(self.fold_mae))

    def se_mae(self, n_splits: int) -> float:
        return float(np.std(self.fold_mae, ddof=1) / np.sqrt(n_splits))


def clip_age(values: np.ndarray, cfg: Config) -> np.ndarray:
    return np.clip(values, 0.0, cfg.evaluation.max_age_clip)


def design(df: pd.DataFrame, candidate: Candidate, cfg: Config) -> pd.DataFrame:
    """Slice to declared inputs only, so a model cannot see any other column."""
    return df[input_columns(cfg.feature_sets[candidate.feature_set])]


def new_model(candidate: Candidate, cfg: Config) -> Any:
    return build_model(candidate, cfg.feature_sets[candidate.feature_set], cfg.seed)


def missingness_shift(aged: pd.DataFrame, unaged: pd.DataFrame, columns: list[str]) -> float:
    if unaged.empty:
        return 0.0
    gaps = [abs(aged[c].isna().mean() - unaged[c].isna().mean()) for c in columns]
    return float(max(gaps, default=0.0))


def cross_validate(dev: pd.DataFrame, cfg: Config) -> dict[str, CandidateCV]:
    y = dev["age"].to_numpy()
    groups = dev["landing_id"].to_numpy()
    results: dict[str, CandidateCV] = {}
    for candidate in cfg.candidates:
        X = design(dev, candidate, cfg)
        fold_mae: list[float] = []
        oof = np.full(len(dev), np.nan)
        for repeat, _fold, train, valid in repeated_group_folds(
            groups, cfg.cv.n_splits, cfg.cv.n_repeats, cfg.seed
        ):
            model = clone(new_model(candidate, cfg)).fit(X.iloc[train], y[train])
            predicted = clip_age(model.predict(X.iloc[valid]), cfg)
            fold_mae.append(float(np.abs(predicted - y[valid]).mean()))
            if repeat == 0:
                oof[valid] = predicted
        results[candidate.name] = CandidateCV(candidate, fold_mae, oof)
    return results


def select_model(
    results: dict[str, CandidateCV], eligible: set[str], n_splits: int
) -> tuple[str, str]:
    """One-standard-error rule: the simplest eligible model within one SE of the best eligible."""
    pool = [r for r in results.values() if r.candidate.name in eligible]
    best = min(pool, key=lambda r: r.mean_mae)
    limit = best.mean_mae + best.se_mae(n_splits)
    chosen = next(r for r in pool if r.mean_mae <= limit)  # pool keeps config order
    return chosen.candidate.name, f"best={best.candidate.name}; one-SE limit MAE={limit:.4f}"


def evaluate_held_out(
    dev: pd.DataFrame, test: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Fit every candidate on all development fish and score held-out landings."""
    y_dev, y_test = dev["age"].to_numpy(), test["age"].to_numpy()
    groups = test["landing_id"].to_numpy()
    rows, predictions, fitted = [], {}, {}
    for candidate in cfg.candidates:
        model = new_model(candidate, cfg).fit(design(dev, candidate, cfg), y_dev)
        predicted = clip_age(model.predict(design(test, candidate, cfg)), cfg)
        ci = cluster_bootstrap_mean(
            np.abs(predicted - y_test), groups, cfg.evaluation.n_bootstrap, cfg.seed
        )
        rows.append(
            {"model": candidate.name, **regression_metrics(y_test, predicted),
             "mae_ci_low": ci["ci_low"], "mae_ci_high": ci["ci_high"]}
        )  # fmt: skip
        predictions[candidate.name] = predicted
        fitted[candidate.name] = model
    return pd.DataFrame(rows), predictions, fitted


def temporal_check(aged: pd.DataFrame, names: list[str], cfg: Config) -> list[dict[str, Any]]:
    """Fit on earlier years and score later ones. Reported only; it does not drive selection."""
    cut = cfg.split.temporal_train_max_year
    past, future = aged[aged["year"] <= cut], aged[aged["year"] > cut]
    if past.empty or future.empty:
        return []
    by_name = {c.name: c for c in cfg.candidates}
    rows = []
    for name in dict.fromkeys(names):
        candidate = by_name[name]
        model = new_model(candidate, cfg).fit(design(past, candidate, cfg), past["age"].to_numpy())
        predicted = clip_age(model.predict(design(future, candidate, cfg)), cfg)
        scored = future.assign(y_true=future["age"], y_pred=predicted)
        seen_gear = scored["gear"].isin(past["gear"].unique()).to_numpy()
        rows.append(
            {
                "model": name,
                "train_years": f"<={cut}",
                "train_n": len(past),
                **regression_metrics(scored["y_true"].to_numpy(), predicted),
                "mae_gears_seen_in_training": float(
                    np.abs(predicted[seen_gear] - scored["y_true"].to_numpy()[seen_gear]).mean()
                ),
            }
        )
    return rows


def package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scikit-learn", "scipy", "pydantic"]
    return {"python": platform.python_version(), **{n: metadata.version(n) for n in names}}


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def run(config_path: str | Path, no_test: bool = False) -> dict[str, Any]:
    """Execute the pipeline. With no_test the held-out landings are never scored."""
    cfg = load_config(config_path)
    df, quality = ingest(cfg)
    aged, unaged = df[df["age"].notna()].copy(), df[df["age"].isna()].copy()
    aged["age"] = aged["age"].astype(float)

    held_out = held_out_landings(aged, cfg.split.salt, cfg.split.test_fraction)
    is_test = aged["landing_id"].isin(held_out)
    dev, test = aged[~is_test].reset_index(drop=True), aged[is_test].reset_index(drop=True)

    run_id = f"{config_hash(config_path)[:8]}-{quality['raw_sha256'][:8]}"
    reports = cfg.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "data_quality.json", quality)

    eligibility: dict[str, str] = {}
    for candidate in cfg.candidates:
        shift = missingness_shift(
            aged, unaged, input_columns(cfg.feature_sets[candidate.feature_set])
        )
        if not candidate.selectable:
            eligibility[candidate.name] = "research_only"
        elif shift > cfg.evaluation.missingness_shift_threshold:
            eligibility[candidate.name] = f"missingness_shift={shift:.2f}"
        else:
            eligibility[candidate.name] = "eligible"
    eligible = {name for name, state in eligibility.items() if state == "eligible"}

    cv = cross_validate(dev, cfg)
    selected, reason = select_model(cv, eligible, cfg.cv.n_splits)
    cv_table = pd.DataFrame(
        [
            {
                "model": name,
                "feature_set": r.candidate.feature_set,
                "status": eligibility[name],
                "cv_mae_mean": r.mean_mae,
                "cv_mae_se": r.se_mae(cfg.cv.n_splits),
                "selected": name == selected,
            }
            for name, r in cv.items()
        ]
    )
    cv_table.to_csv(reports / "cv_results.csv", index=False)

    split_summary = {
        "dev_fish": len(dev),
        "dev_landings": int(dev["landing_id"].nunique()),
        "test_fish": len(test),
        "test_landings": int(test["landing_id"].nunique()),
        "test_gear_counts": test["gear"].value_counts().to_dict(),
    }
    summary: dict[str, Any] = {
        "run_id": run_id,
        "selected_model": selected,
        "selection_note": reason,
        "split": split_summary,
    }
    if no_test:
        summary["test_evaluated"] = False
        return summary

    conformal = MondrianConformal(
        cfg.conformal.alpha, cfg.conformal.predicted_age_edges, cfg.conformal.min_bin_size
    ).fit(dev["age"].to_numpy(), cv[selected].oof_first_repeat)

    test_table, predictions, fitted = evaluate_held_out(dev, test, cfg)
    test_table.to_csv(reports / "test_results.csv", index=False)

    y_test, groups = test["age"].to_numpy(), test["landing_id"].to_numpy()
    reference = cfg.promotion.reference_model
    selected_pred, reference_pred = predictions[selected], predictions[reference]
    delta = cluster_bootstrap_mean(
        np.abs(selected_pred - y_test) - np.abs(reference_pred - y_test),
        groups, cfg.evaluation.n_bootstrap, cfg.seed,
    )  # fmt: skip
    lower, upper = conformal.interval(selected_pred)
    covered = ((y_test >= lower) & (y_test <= upper)).astype(float)
    coverage = cluster_bootstrap_mean(covered, groups, cfg.evaluation.n_bootstrap, cfg.seed)

    scored = test.assign(y_true=y_test, y_pred=selected_pred)
    scored["age_band"] = pd.cut(scored["y_true"], AGE_BINS, labels=AGE_LABELS)
    slices = pd.concat(
        [slice_table(scored, by) for by in ("age_band", "gear", "scale_readability", "sex")],
        ignore_index=True,
    )
    slices.to_csv(reports / "test_slices.csv", index=False)

    selected_candidate = cv[selected].candidate
    importance = permutation_importance(
        fitted[selected], design(test, selected_candidate, cfg), y_test,
        scoring="neg_mean_absolute_error", n_repeats=20, random_state=cfg.seed,
    )  # fmt: skip
    importances = dict(
        zip(
            design(test, selected_candidate, cfg).columns,
            importance.importances_mean.round(4).tolist(),
            strict=True,
        )
    )
    temporal = temporal_check(aged, [selected, reference], cfg)

    beats_reference = selected == reference or delta["ci_high"] < 0
    covers = coverage["estimate"] >= cfg.promotion.min_interval_coverage
    promoted = bool(beats_reference and covers and selected != "dummy_median")

    # The final artefact is refit on every aged fish. The test set has been used by this point and
    # no decision can change, so the reported metrics describe the procedure, as is standard.
    final_model = new_model(selected_candidate, cfg).fit(
        design(aged, selected_candidate, cfg), aged["age"].to_numpy()
    )
    bundle = {
        "model": final_model,
        "conformal": conformal,
        "model_name": selected,
        "input_columns": list(design(aged, selected_candidate, cfg).columns),
        "max_age_clip": cfg.evaluation.max_age_clip,
        "length_range_mm": (float(aged["total_length_mm"].min()),
                            float(aged["total_length_mm"].max())),
        "reference": {c: aged[c].to_numpy(dtype=float) for c in DRIFT_COLUMNS},
        "run_id": run_id,
    }  # fmt: skip
    model_dir = cfg.paths.models_dir / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = model_dir / "bundle.joblib"
    joblib.dump(bundle, bundle_path)

    results = {
        **summary,
        "test_evaluated": True,
        "promoted": promoted,
        "promotion_checks": {
            "reference_model": reference,
            "mae_difference_vs_reference": delta,
            "beats_reference": bool(beats_reference),
            "interval_coverage": coverage,
            "target_coverage": 1 - cfg.conformal.alpha,
            "coverage_ok": bool(covers),
        },
        "test_metrics_selected": test_table.set_index("model").loc[selected].to_dict(),
        "conformal_half_widths": dict(
            zip(["<4", "4-7", "7-10", ">=10"], conformal.half_widths or [], strict=False)
        ),
        "permutation_importance_mae": importances,
        "temporal_check": temporal,
        "unaged_population": {
            "n": len(unaged),
            "length_mean_aged": float(aged["total_length_mm"].mean()),
            "length_mean_unaged": float(unaged["total_length_mm"].mean()),
            "length_outside_aged_range": int(
                (~unaged["total_length_mm"].between(
                    aged["total_length_mm"].min(), aged["total_length_mm"].max())).sum()
            ),
            "readability_counts": unaged["scale_readability"].value_counts(dropna=False).to_dict(),
        },
    }  # fmt: skip
    write_json(reports / "metrics.json", results)
    write_json(
        model_dir / "manifest.json",
        {
            "run_id": run_id,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "model_name": selected,
            "promoted": promoted,
            "bundle_sha256": sha256_file(bundle_path),
            "raw_data_sha256": quality["raw_sha256"],
            "config_sha256": config_hash(config_path),
            "git_commit": git_commit(),
            "seed": cfg.seed,
            "versions": package_versions(),
        },
    )
    pointer = cfg.paths.models_dir / "production.json"
    if promoted:
        write_json(pointer, {"run_id": run_id, "bundle": str(bundle_path)})
    else:
        pointer.unlink(missing_ok=True)

    plot_model_comparison(test_table, selected, reports / "model_comparison.png")
    plot_calibration(y_test, selected_pred, lower, upper, reports / "calibration_by_age.png")
    write_model_card(reports / "model_card.md", results, quality, test_table, slices)
    return results
