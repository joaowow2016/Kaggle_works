import numpy as np
import pytest

from pagrus_age.metrics import MondrianConformal, cluster_bootstrap_mean, psi, regression_metrics


def test_regression_metrics_by_hand():
    result = regression_metrics(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 3.0, 3.0, 7.0]))
    assert result["mae"] == pytest.approx(1.0)
    assert result["bias"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["exact_rate"] == pytest.approx(0.5)
    assert result["within_1yr_rate"] == pytest.approx(0.75)


def test_bootstrap_estimate_weights_by_fish():
    values = np.array([0.0, 0.0, 0.0, 4.0])
    groups = np.array(["a", "a", "a", "b"])
    result = cluster_bootstrap_mean(values, groups, n_boot=500, seed=0)
    assert result["estimate"] == pytest.approx(1.0)
    assert result["ci_low"] <= 1.0 <= result["ci_high"]


def test_bootstrap_widens_when_fish_within_a_landing_are_correlated():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 50)
    clustered = np.repeat(rng.normal(size=20), 50)
    independent = np.repeat(rng.normal(size=1000), 1)

    def width(values, group_ids):
        ci = cluster_bootstrap_mean(values, group_ids, 500, 0)
        return ci["ci_high"] - ci["ci_low"]

    assert width(clustered, groups) > 2 * width(independent, np.arange(1000))


def test_conformal_reaches_target_coverage_on_exchangeable_data():
    rng = np.random.default_rng(1)
    predicted = rng.uniform(1, 15, 8000)
    truth = predicted + rng.normal(0, 0.3 + 0.1 * predicted)
    model = MondrianConformal(0.1, [4, 7, 10], 50).fit(truth[:4000], predicted[:4000])
    lower, upper = model.interval(predicted[4000:])
    coverage = ((truth[4000:] >= lower) & (truth[4000:] <= upper)).mean()
    assert coverage == pytest.approx(0.9, abs=0.02)
    assert model.half_widths == sorted(model.half_widths)


def test_conformal_small_bin_uses_overall_width():
    predicted = np.concatenate([np.full(200, 2.0), np.full(3, 12.0)])
    truth = predicted + np.linspace(-1, 1, 203)
    model = MondrianConformal(0.1, [4, 7, 10], 50).fit(truth, predicted)
    assert model.half_widths[3] == model._quantile(np.abs(truth - predicted), 0.1)


def test_psi_is_near_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=5000)
    edges = np.quantile(reference, np.linspace(0.1, 0.9, 9))
    assert psi(reference, rng.normal(size=5000), edges) < 0.02
    assert psi(reference, rng.normal(2, 1, 5000), edges) > 0.25
