"""MLflow is an optional extra, so these skip without it."""

from __future__ import annotations

import pytest

mlflow = pytest.importorskip("mlflow")

from gev.tracking import (  # noqa: E402
    RUN_ID_FILE,
    MlflowTracker,
    Tracker,
    flat_params,
    log_calibration,
    start,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "gev-test")
    yield tmp_path / "out"
    mlflow.end_run()


def test_disabled_tracking_is_a_no_op(tmp_path):
    tracker = start(tmp_path, enabled=False)
    assert type(tracker) is Tracker
    assert not (tmp_path / RUN_ID_FILE).exists()


def test_a_resumed_run_appends_to_the_run_that_trained_it(store):
    first = MlflowTracker(store, system_metrics=False)
    first.log_params({"learning_rate": "5e-05"})
    first.log_metrics({"loss/label_token": 1.0}, step=25)
    first.finish()

    second = MlflowTracker(store, system_metrics=False)
    # A changed value would raise in MLflow if it were logged again.
    second.log_params({"learning_rate": "1e-04"})
    second.log_metrics({"loss/label_token": 0.5}, step=50)
    second.finish()

    assert second.run_id == first.run_id and second.resumed
    run = mlflow.get_run(first.run_id)
    assert run.data.params["learning_rate"] == "5e-05"
    history = mlflow.MlflowClient().get_metric_history(first.run_id, "loss/label_token")
    assert [(m.step, m.value) for m in history] == [(25, 1.0), (50, 0.5)]


def test_calibration_lands_on_the_training_run(store):
    tracker = MlflowTracker(store, system_metrics=False)
    profile = store / "calibration.json"
    profile.write_text("{}")
    log_calibration(tracker, {"temperatures": {"choice:label_token:2-4": 1.3}, "profile": profile})
    tracker.finish()

    run = mlflow.get_run(tracker.run_id)
    assert run.data.metrics == {"calibration/T/choice_label_token_2-4": 1.3}
    artifacts = [a.path for a in mlflow.MlflowClient().list_artifacts(tracker.run_id)]
    assert artifacts == ["calibration.json"], "a missing report is skipped, not an error"


def test_config_flattens_to_string_params():
    from gev.train.config import PRESETS

    params = flat_params(PRESETS["e4b-instruct-2xa30"])
    assert params["gpu.name"] == "A30 24GB" and params["n_gpus"] == "2"


def test_a_fresh_training_run_opens_a_new_mlflow_run(store, monkeypatch):
    """`--fresh` sets the run id aside with the checkpoints, before any tracker reads it."""
    from dataclasses import replace

    import gev.train.loop as loop
    from gev.train.config import PRESETS

    trackers = []

    def train(config, *args):
        trackers.append(MlflowTracker(config.output_dir, system_metrics=False))
        trackers[-1].finish()
        return {}

    monkeypatch.setattr(loop, "_train", train)
    config = replace(PRESETS["smoke"], output_dir=str(store))
    loop.run_training(config, "unused")
    loop.run_training(config, "unused")
    loop.run_training(config, "unused", fresh=True)

    first, resumed, fresh = trackers
    assert resumed.run_id == first.run_id, "a plain rerun resumes its run"
    assert fresh.run_id != first.run_id and not fresh.resumed


def test_a_release_uploads_to_the_run_that_trained_it(store, tmp_path):
    from gev import cli
    from test_release import fake_checkpoint

    out = fake_checkpoint(tmp_path)
    training = MlflowTracker(out, system_metrics=False)
    training.finish()

    cli.main(["release", "build", "--checkpoint", str(out), "--out", str(tmp_path / "rel")])
    assert not mlflow.MlflowClient().list_artifacts(training.run_id), "only with --mlflow"
    cli.main(
        ["release", "build", "--checkpoint", str(out), "--out", str(tmp_path / "rel"), "--mlflow"]
    )

    client = mlflow.MlflowClient()
    files = {a.path for a in client.list_artifacts(training.run_id, "release")}
    assert {"release/gev_release.json", "release/adapter_model.safetensors"} <= files
    assert "release/training_state.pt" not in files


def test_a_release_without_a_training_run_fails_before_copying(store, tmp_path):
    from gev import cli
    from test_release import fake_checkpoint

    out = fake_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match="trained without --mlflow"):
        cli.main(
            [
                "release",
                "build",
                "--checkpoint",
                str(out),
                "--out",
                str(tmp_path / "rel"),
                "--mlflow",
            ]
        )
    assert not (tmp_path / "rel").exists()
