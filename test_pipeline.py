"""End-to-end checks on synthetic data: artefacts, gate, reproducibility, inference and API."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pagrus_age.cli import main
from pagrus_age.data import DataContractError
from pagrus_age.predict import (
    ModelNotPromotedError,
    drift_report,
    load_bundle,
    predict_file,
    predict_frame,
)
from pagrus_age.serve import create_app
from pagrus_age.train import run


@pytest.fixture
def trained(project):
    cfg, path = project
    return cfg, path, run(path)


def test_run_writes_every_artefact(trained):
    cfg, _, results = trained
    for name in ("metrics.json", "cv_results.csv", "test_results.csv", "test_slices.csv",
                 "data_quality.json", "model_card.md", "model_comparison.png",
                 "calibration_by_age.png"):  # fmt: skip
        assert (cfg.paths.reports_dir / name).exists(), name
    manifest = json.loads((cfg.paths.models_dir / results["run_id"] / "manifest.json").read_text())
    assert manifest["versions"]["scikit-learn"]


def test_research_model_cannot_be_selected(trained):
    cfg, _, results = trained
    assert results["selected_model"] != "hgb_dissection"
    table = pd.read_csv(cfg.paths.reports_dir / "cv_results.csv").set_index("model")
    assert table.loc["hgb_dissection", "status"] == "research_only"


def test_no_test_mode_touches_no_test_artefacts(project):
    cfg, path = project
    results = run(path, no_test=True)
    assert results["test_evaluated"] is False
    assert not (cfg.paths.reports_dir / "test_results.csv").exists()


def test_rerun_gives_identical_metrics(project):
    _, path = project
    first = run(path)
    second = run(path)
    assert first["test_metrics_selected"] == second["test_metrics_selected"]


def test_contract_violation_stops_training(project):
    cfg, path = project
    frame = pd.read_csv(cfg.paths.raw_csv).rename(columns={"Age": "age_years"})
    frame.to_csv(cfg.paths.raw_csv, index=False)
    with pytest.raises(DataContractError):
        run(path)


def test_unpromoted_run_is_refused_unless_named(trained):
    cfg, _, results = trained
    (cfg.paths.models_dir / "production.json").unlink(missing_ok=True)
    with pytest.raises(ModelNotPromotedError):
        load_bundle(cfg.paths.models_dir)
    assert load_bundle(cfg.paths.models_dir, results["run_id"])["run_id"] == results["run_id"]


def test_tampered_bundle_is_refused(trained):
    cfg, _, results = trained
    path = cfg.paths.models_dir / results["run_id"] / "bundle.joblib"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="does not match"):
        load_bundle(cfg.paths.models_dir, results["run_id"])


def test_predict_needs_only_length(trained):
    cfg, _, results = trained
    bundle = load_bundle(cfg.paths.models_dir, results["run_id"])
    out = predict_frame(bundle, pd.DataFrame({"total_length_mm": [150.0, 340.0, 3000.0]}))
    assert out["age_pred"].is_monotonic_increasing
    assert (out["age_lower_90"] <= out["age_pred"]).all()
    assert (out["age_upper_90"] >= out["age_pred"]).all()
    assert out["length_outside_training_range"].tolist() == [False, False, True]


def test_predict_rejects_missing_length(trained):
    cfg, _, results = trained
    bundle = load_bundle(cfg.paths.models_dir, results["run_id"])
    with pytest.raises(ValueError, match="required"):
        predict_frame(bundle, pd.DataFrame({"total_length_mm": [None]}, dtype=float))


def test_batch_file_predicts_only_unaged_fish(trained, tmp_path):
    cfg, _, results = trained
    bundle = load_bundle(cfg.paths.models_dir, results["run_id"])
    raw = pd.read_csv(cfg.paths.raw_csv)
    raw.loc[raw["Age"].notna().idxmax(), "Scale_readability"] = "unreadable"
    raw.to_csv(cfg.paths.raw_csv, index=False)
    out = predict_file(bundle, cfg.paths.raw_csv, tmp_path / "out.csv", cfg.qc, unaged_only=True)
    # The same QC as training applies: an age on an unreadable scale counts as unaged.
    assert len(out) == raw["Age"].isna().sum() + 1


def test_drift_alerts_on_shifted_batch(trained):
    cfg, _, results = trained
    bundle = load_bundle(cfg.paths.models_dir, results["run_id"])
    processed = pd.read_parquet(cfg.paths.processed)
    assert not any("total_length_mm" in a for a in drift_report(bundle, processed)["alerts"])
    shifted = processed.assign(total_length_mm=processed["total_length_mm"] * 1.5)
    assert any("total_length_mm" in a for a in drift_report(bundle, shifted)["alerts"])


def test_api(trained):
    cfg, _, results = trained
    app = create_app(str(cfg.paths.models_dir), results["run_id"])
    with TestClient(app) as client:
        assert client.get("/health").json()["run_id"] == results["run_id"]
        ok = client.post("/predict", json={"fish": [{"total_length_mm": 340, "month": 8}]})
        assert ok.status_code == 200
        assert ok.json()["predictions"][0]["age_lower_90"] >= 0
        assert client.post("/predict", json={"fish": [{"month": 8}]}).status_code == 422
        bad = client.post("/predict", json={"fish": [{"total_length_mm": 340, "age": 5}]})
        assert bad.status_code == 422


def test_cli_train_no_test(project, capsys):
    _, path = project
    assert main(["--config", str(path), "train", "--no-test"]) == 0
    assert json.loads(capsys.readouterr().out)["test_evaluated"] is False
