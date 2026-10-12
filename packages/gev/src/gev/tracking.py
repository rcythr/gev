"""Experiment tracking: MLflow when asked for, otherwise nothing.

The MLflow server and experiment come from `MLFLOW_TRACKING_URI` and
`MLFLOW_EXPERIMENT_NAME` (default `gev`); without a URI, MLflow writes locally.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path

RUN_ID_FILE = "mlflow_run_id"
DEFAULT_EXPERIMENT = "gev"


class Tracker:
    """The no-op tracker; also what non-main ranks get."""

    def log_params(self, params: dict) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_artifact(self, path: str | Path) -> None:
        pass

    def finish(self, status: str = "FINISHED") -> None:
        pass


class MlflowTracker(Tracker):
    """One MLflow run per output directory.

    The run id is kept beside the checkpoints, so a resumed run and the later
    calibration append to the run that trained the weights instead of starting new ones.
    """

    def __init__(self, output_dir: str | Path, *, system_metrics: bool = True):
        import mlflow

        self.mlflow = mlflow
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        id_file = output / RUN_ID_FILE
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))
        run_id = id_file.read_text().strip() if id_file.is_file() else None
        self.resumed = run_id is not None
        run = mlflow.start_run(
            run_id=run_id,
            run_name=None if run_id else output.name,
            log_system_metrics=system_metrics,
        )
        if not self.resumed:
            id_file.write_text(run.info.run_id + "\n")
        self.run_id = run.info.run_id

    def log_params(self, params: dict) -> None:
        # MLflow refuses to change a logged param, and a resumed run's were logged at its start.
        if not self.resumed:
            self.mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if Path(path).is_file():
            self.mlflow.log_artifact(str(path))

    def finish(self, status: str = "FINISHED") -> None:
        self.mlflow.end_run(status=status)


def start(output_dir: str | Path, enabled: bool) -> Tracker:
    if not enabled:
        return Tracker()
    try:
        return MlflowTracker(output_dir)
    except ModuleNotFoundError as error:
        raise RuntimeError("--mlflow needs the mlflow extra: uv sync --extra mlflow") from error


def flat_params(config) -> dict[str, str]:
    """A config as MLflow params: nested dataclasses dotted, values as strings."""
    out: dict[str, str] = {}
    for key, value in (asdict(config) if is_dataclass(config) else config).items():
        if isinstance(value, dict):
            out.update({f"{key}.{k}": str(v) for k, v in value.items()})
        else:
            out[key] = str(value)
    return out


def log_calibration(tracker: Tracker, result: dict) -> None:
    """Fitted temperatures as metrics, and the profile and its report as artifacts."""
    # MLflow metric names allow only word characters and `./ -`.
    tracker.log_metrics(
        {
            "calibration/T/" + re.sub(r"[^\w./ -]", "_", bucket): t
            for bucket, t in result["temperatures"].items()
        }
    )
    profile = Path(result["profile"])
    tracker.log_artifact(profile)
    tracker.log_artifact(profile.with_name("calibration.report.json"))
