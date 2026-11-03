import numpy as np
import pandas as pd
import pytest

from pagrus_age.config import Candidate, FeatureSet, load_config
from pagrus_age.data import to_canonical
from pagrus_age.models import (
    FORBIDDEN_FEATURES,
    AgeLengthKey,
    InverseVonBertalanffy,
    add_derived,
    assert_no_forbidden,
    build_model,
    input_columns,
)
from pagrus_age.splits import held_out_landings, repeated_group_folds

from .conftest import ROOT


@pytest.fixture
def fish(raw_frame):
    frame = to_canonical(raw_frame)
    return frame[frame["age"].notna()].reset_index(drop=True)


def test_test_landings_do_not_depend_on_row_order(fish):
    shuffled = fish.sample(frac=1, random_state=1)
    assert held_out_landings(fish, "s", 0.2) == held_out_landings(shuffled, "s", 0.2)


def test_every_gear_reaches_the_test_set(fish):
    chosen = held_out_landings(fish, "s", 0.2)
    assert set(fish[fish["landing_id"].isin(chosen)]["gear"]) == set(fish["gear"])


def test_salt_changes_the_split(fish):
    assert held_out_landings(fish, "a", 0.3) != held_out_landings(fish, "b", 0.3)


def test_folds_never_share_a_landing(fish):
    groups = fish["landing_id"].to_numpy()
    for _, _, train, valid in repeated_group_folds(groups, 5, 2, seed=0):
        assert not set(groups[train]) & set(groups[valid])


def test_shipped_config_uses_no_forbidden_feature():
    cfg = load_config(ROOT / "configs/default.yaml")
    for features in cfg.feature_sets.values():
        assert_no_forbidden(features)
        assert not set(input_columns(features)) & FORBIDDEN_FEATURES


@pytest.mark.parametrize("column", ["age", "scale_border", "scale_readability", "condition_factor"])
def test_leaky_feature_is_refused(column):
    features = FeatureSet(numeric=["total_length_mm", column], categorical=[])
    candidate = Candidate(name="x", kind="hgb", feature_set="f")
    with pytest.raises(ValueError, match="forbidden"):
        build_model(candidate, features, seed=0)


def test_month_encoding_is_cyclic():
    out = add_derived(pd.DataFrame({"month": [12, 1, 6]}))
    distance = np.hypot(*(out.loc[0, ["month_sin", "month_cos"]].to_numpy(float)
                          - out.loc[1, ["month_sin", "month_cos"]].to_numpy(float)))  # fmt: skip
    far = np.hypot(*(out.loc[0, ["month_sin", "month_cos"]].to_numpy(float)
                     - out.loc[2, ["month_sin", "month_cos"]].to_numpy(float)))  # fmt: skip
    assert distance < far


def test_age_length_key_averages_a_bin_and_borrows_for_empty_bins():
    X = pd.DataFrame({"total_length_mm": [200.0, 210.0, 400.0]})
    model = AgeLengthKey(bin_mm=20).fit(X, np.array([2.0, 4.0, 10.0]))
    predicted = model.predict(pd.DataFrame({"total_length_mm": [205.0, 395.0, 900.0]}))
    assert predicted.tolist() == [3.0, 10.0, 10.0]


def test_von_bertalanffy_recovers_known_curve():
    age = np.arange(1, 20, dtype=float).repeat(5)
    length = 450 * (1 - np.exp(-0.2 * (age + 1)))
    model = InverseVonBertalanffy().fit(pd.DataFrame({"total_length_mm": length}), age)
    assert model.l_inf_ == pytest.approx(450, rel=1e-3)
    predicted = model.predict(pd.DataFrame({"total_length_mm": length}))
    assert np.abs(predicted[:60] - age[:60]).max() < 0.05


def test_length_monotonicity_is_enforced_in_boosting(fish):
    cfg = load_config(ROOT / "configs/default.yaml")
    candidate = next(c for c in cfg.candidates if c.name == "hgb_dock_small")
    features = cfg.feature_sets["dock"]
    model = build_model(candidate, features, seed=0)
    model.fit(fish[input_columns(features)], fish["age"].to_numpy())
    grid = pd.concat([fish[input_columns(features)].iloc[[0]]] * 60, ignore_index=True)
    grid["total_length_mm"] = np.linspace(100, 600, 60)
    assert (np.diff(model.predict(grid)) >= -1e-9).all()
