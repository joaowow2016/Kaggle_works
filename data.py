"""Download, data contract, canonicalisation and quality control.

The contract is strict on purpose: a changed file should stop the pipeline, not train on it.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pagrus_age.config import QC, Config

# "Total_lenght" is how the publisher spells it.
RAW_COLUMNS = [
    "ID", "Landing_acronym", "Day", "Month", "Year", "Fishing_gear", "Min_depth", "Max_depth",
    "Sex", "Total_lenght", "Total_weight", "Gonad_weight", "Maturity_stage", "Condition_factor",
    "Age", "Scale_readability", "Scale_border",
]  # fmt: skip
GEARS = {"Hand_line_fishing_boat": "handline", "Otter_board_trawler": "otter_trawl",
         "Pair_trawler": "pair_trawl"}  # fmt: skip
SEX_CODES = {1: "male", 2: "female", 3: "undetermined", 4: "hermaphrodite"}
READABILITY = {"Well_marked": "well_marked", "Faint_or_absent": "faint_or_absent",
               "unreadable": "unreadable"}  # fmt: skip
TRIP_LEVEL_COLUMNS = ["Fishing_gear", "Year", "Month", "Day", "Min_depth", "Max_depth"]


class DataContractError(ValueError):
    """The raw file does not match what the pipeline was built and validated against."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(cfg: Config) -> Path:
    """Fetch the SEANOE file unless present, and verify its pinned checksum either way."""
    target = cfg.paths.raw_csv
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        request = urllib.request.Request(cfg.source.url, headers={"User-Agent": "pagrus-age/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(target)
    actual = sha256_file(target)
    if actual != cfg.source.sha256:
        raise DataContractError(f"{target}: sha256 {actual} != pinned {cfg.source.sha256}")
    return target


def read_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, skipinitialspace=True, na_values=["", " "])


def validate_raw(raw: pd.DataFrame) -> None:
    """Raise DataContractError listing every violation found."""
    if list(raw.columns) != RAW_COLUMNS:
        raise DataContractError(f"columns differ from contract: {list(raw.columns)}")
    problems: list[str] = []
    if raw["ID"].duplicated().any():
        problems.append("duplicated ID")
    required = ["ID", "Landing_acronym", "Day", "Month", "Year", "Fishing_gear", "Total_lenght"]
    for column in required:
        if raw[column].isna().any():
            problems.append(f"missing values in {column}")
    if unknown := set(raw["Fishing_gear"].dropna()) - set(GEARS):
        problems.append(f"unknown gear {sorted(unknown)}")
    if unknown := set(raw["Sex"].dropna()) - set(SEX_CODES):
        problems.append(f"unknown sex code {sorted(unknown)}")
    if unknown := set(raw["Scale_readability"].dropna()) - set(READABILITY):
        problems.append(f"unknown readability {sorted(unknown)}")
    if (raw["Min_depth"] > raw["Max_depth"]).any():
        problems.append("Min_depth > Max_depth")
    dates = pd.to_datetime(
        {"year": 1900 + raw["Year"], "month": raw["Month"], "day": raw["Day"]}, errors="coerce"
    )
    if dates.isna().any():
        problems.append(f"{int(dates.isna().sum())} invalid dates")
    per_landing = raw.groupby("Landing_acronym")[TRIP_LEVEL_COLUMNS].nunique(dropna=False)
    if (per_landing > 1).any(axis=None):
        problems.append("a landing has more than one date, gear or depth range")
    if problems:
        raise DataContractError("; ".join(problems))


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename and recode only. No rows are dropped and no values are judged here."""
    maturity = raw["Maturity_stage"].astype("Float64").round().astype("Int64")
    return pd.DataFrame(
        {
            "fish_id": raw["ID"].astype(str),
            "landing_id": raw["Landing_acronym"].astype(str),
            "date": pd.to_datetime(
                {"year": 1900 + raw["Year"], "month": raw["Month"], "day": raw["Day"]}
            ),
            "year": (1900 + raw["Year"]).astype(int),
            "month": raw["Month"].astype(int),
            "gear": raw["Fishing_gear"].map(GEARS).astype(object),
            "depth_min_m": raw["Min_depth"].astype(float),
            "depth_max_m": raw["Max_depth"].astype(float),
            "sex": raw["Sex"].map(SEX_CODES).astype(object),
            "maturity_stage": maturity.map(
                lambda v: f"stage_{v}" if pd.notna(v) else np.nan
            ).astype(object),
            "total_length_mm": raw["Total_lenght"].astype(float),
            "total_weight_g": raw["Total_weight"].astype(float),
            "gonad_weight_g": raw["Gonad_weight"].astype(float),
            "condition_factor": raw["Condition_factor"].astype(float),
            "age": raw["Age"].astype(float),
            "scale_readability": raw["Scale_readability"].map(READABILITY).astype(object),
            "scale_border": raw["Scale_border"].astype(float),
        }
    )


def apply_qc(df: pd.DataFrame, qc: QC) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply fixed-rule QC. Suspect values become missing so that the rest of the fish survives."""
    out = df.copy()
    log: dict[str, Any] = {"rows_in": len(out)}

    keep = out["total_length_mm"].between(*qc.length_mm)
    log["dropped_length_out_of_range"] = int((~keep).sum())
    out = out[keep].copy()

    bad_age = out["age"].notna() & ~out["age"].between(*qc.age_years)
    # An age on a scale its own record calls unreadable contradicts itself.
    contradictory = out["age"].notna() & out["scale_readability"].eq("unreadable")
    log["age_out_of_range_set_missing"] = int(bad_age.sum())
    log["age_on_unreadable_scale_set_missing"] = int(contradictory.sum())
    out.loc[bad_age | contradictory, "age"] = np.nan

    bad_weight = out["total_weight_g"].le(0)
    expected = qc.allometry_a * out["total_length_mm"] ** qc.allometry_b
    residual = np.log(out["total_weight_g"].where(~bad_weight) / expected)
    outlier = residual.abs() > qc.max_log_residual
    log["weight_allometry_outlier_set_missing"] = int(outlier.sum())
    out["weight_qc_flag"] = outlier
    out.loc[bad_weight | outlier, "total_weight_g"] = np.nan

    bad_gonad = out["gonad_weight_g"] >= out["total_weight_g"]
    log["gonad_heavier_than_body_set_missing"] = int(bad_gonad.sum())
    out.loc[bad_gonad, "gonad_weight_g"] = np.nan

    log["rows_out"] = len(out)
    log["aged_fish"] = int(out["age"].notna().sum())
    log["unaged_fish"] = int(out["age"].isna().sum())
    log["landings"] = int(out["landing_id"].nunique())
    log["landings_with_aged_fish"] = int(out.loc[out["age"].notna(), "landing_id"].nunique())
    return out.reset_index(drop=True), log


def ingest(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = read_raw(cfg.paths.raw_csv)
    validate_raw(raw)
    clean, log = apply_qc(to_canonical(raw), cfg.qc)
    log["raw_sha256"] = sha256_file(cfg.paths.raw_csv)
    cfg.paths.processed.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(cfg.paths.processed, index=False)
    return clean, log
