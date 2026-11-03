import numpy as np
import pandas as pd
import pytest

from pagrus_age.config import QC
from pagrus_age.data import DataContractError, apply_qc, to_canonical, validate_raw

QC_RULES = QC(length_mm=(50, 800), age_years=(0, 30), allometry_a=1.735e-5, allometry_b=2.979,
              max_log_residual=0.5)  # fmt: skip


def test_valid_fixture_passes(raw_frame):
    validate_raw(raw_frame)


def test_renamed_column_is_rejected(raw_frame):
    with pytest.raises(DataContractError, match="columns differ"):
        validate_raw(raw_frame.rename(columns={"Total_lenght": "Total_length"}))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Fishing_gear", "Drone", "unknown gear"),
        ("Sex", 9.0, "unknown sex"),
        ("Scale_readability", "blurry", "unknown readability"),
        ("Month", 13, "invalid dates"),
        ("Min_depth", 999.0, "Min_depth > Max_depth"),
    ],
)
def test_bad_values_are_rejected(raw_frame, column, value, message):
    raw_frame.loc[0, column] = value
    with pytest.raises(DataContractError, match=message):
        validate_raw(raw_frame)


def test_duplicate_id_is_rejected(raw_frame):
    raw_frame.loc[1, "ID"] = raw_frame.loc[0, "ID"]
    with pytest.raises(DataContractError, match="duplicated ID"):
        validate_raw(raw_frame)


def test_landing_with_two_gears_is_rejected(raw_frame):
    raw_frame.loc[0, "Fishing_gear"] = (
        "Pair_trawler"
        if raw_frame.loc[0, "Fishing_gear"] != "Pair_trawler"
        else "Otter_board_trawler"
    )
    with pytest.raises(DataContractError, match="more than one"):
        validate_raw(raw_frame)


def test_canonical_decodes_values(raw_frame):
    raw_frame.loc[0, ["Year", "Sex"]] = [76, 4.0]
    canonical = to_canonical(raw_frame)
    assert canonical.loc[0, "year"] == 1976
    assert canonical.loc[0, "sex"] == "hermaphrodite"
    assert len(canonical) == len(raw_frame)


def test_qc_nulls_implausible_weight_but_keeps_the_fish(raw_frame):
    canonical = to_canonical(raw_frame)
    canonical.loc[0, ["total_length_mm", "total_weight_g"]] = [265.0, 2000.0]
    clean, log = apply_qc(canonical, QC_RULES)
    assert np.isnan(clean.loc[0, "total_weight_g"])
    assert bool(clean.loc[0, "weight_qc_flag"])
    assert log["rows_out"] == log["rows_in"]


def test_qc_removes_age_from_unreadable_scale(raw_frame):
    canonical = to_canonical(raw_frame)
    canonical.loc[0, ["age", "scale_readability"]] = [5.0, "unreadable"]
    clean, log = apply_qc(canonical, QC_RULES)
    assert pd.isna(clean.loc[0, "age"])
    assert log["age_on_unreadable_scale_set_missing"] >= 1


def test_qc_drops_impossible_length(raw_frame):
    canonical = to_canonical(raw_frame)
    canonical.loc[0, "total_length_mm"] = 5000.0
    clean, log = apply_qc(canonical, QC_RULES)
    assert log["dropped_length_out_of_range"] == 1
    assert len(clean) == len(canonical) - 1
