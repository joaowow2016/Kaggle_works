"""Synthetic fixture in the publisher's schema. Tests never depend on the real file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pagrus_age.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
GEARS = ["Otter_board_trawler", "Pair_trawler", "Hand_line_fishing_boat"]


def make_raw(n_landings: int = 40, per_landing: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    fish = 0
    for landing in range(1, n_landings + 1):
        gear = GEARS[landing % 3]
        year = 76 + landing % 9
        month, day = int(rng.integers(1, 13)), int(rng.integers(1, 28))
        depth_min = float(rng.integers(10, 60))
        depth_max = depth_min + float(rng.integers(5, 60))
        for sequence in range(1, per_landing + 1):
            fish += 1
            age = int(np.clip(rng.gamma(5, 1.2), 1, 21))
            length = 447 * (1 - np.exp(-0.204 * (age + 1.134))) + rng.normal(0, 25)
            weight = 1.735e-5 * length**2.979 * np.exp(rng.normal(0, 0.1))
            aged = rng.random() > 0.2
            rows.append(
                {
                    "ID": f"{fish}_{landing}_{sequence}",
                    "Landing_acronym": landing,
                    "Day": day,
                    "Month": month,
                    "Year": year,
                    "Fishing_gear": gear,
                    "Min_depth": depth_min,
                    "Max_depth": depth_max,
                    "Sex": float(rng.integers(1, 5)) if rng.random() > 0.5 else np.nan,
                    "Total_lenght": int(max(length, 100)),
                    "Total_weight": round(weight, 1) if rng.random() > 0.4 else np.nan,
                    "Gonad_weight": 5.0 if rng.random() > 0.8 else np.nan,
                    "Maturity_stage": float(rng.choice([1, 2, 3, 7]))
                    if rng.random() > 0.5
                    else np.nan,
                    "Condition_factor": 1.0,
                    "Age": float(age) if aged else np.nan,
                    "Scale_readability": "Well_marked" if aged else "unreadable",
                    "Scale_border": 2.0 if aged else np.nan,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def project(tmp_path: Path) -> tuple[Config, Path]:
    """A config pointing at a synthetic CSV, with paths inside a temp directory."""
    settings = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    raw_path = tmp_path / "raw.csv"
    make_raw().to_csv(raw_path, index=False)
    settings["paths"] = {
        "raw_csv": str(raw_path),
        "processed": str(tmp_path / "fish.parquet"),
        "reports_dir": str(tmp_path / "reports"),
        "models_dir": str(tmp_path / "models"),
    }
    settings["cv"] = {"n_splits": 3, "n_repeats": 1}
    settings["evaluation"]["n_bootstrap"] = 100
    keep = {"dummy_median", "alk_20mm", "ridge_length", "hgb_dock_small", "hgb_dissection"}
    settings["candidates"] = [c for c in settings["candidates"] if c["name"] in keep]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return load_config(config_path), config_path
