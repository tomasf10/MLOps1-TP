from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


@dataclass(frozen=True)
class Settings:
    dataset_path: Path = _path_from_env(
        "DATASET_PATH", PROJECT_ROOT / "data/raw/diabetes_dataset.csv"
    )
    model_output_path: Path = _path_from_env(
        "MODEL_OUTPUT_PATH", PROJECT_ROOT / "artifacts/model.joblib"
    )
    metrics_output_path: Path = _path_from_env(
        "METRICS_OUTPUT_PATH", PROJECT_ROOT / "artifacts/metrics.json"
    )
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow_experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "diabetes-tp-mlops"
    )
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))


settings = Settings()
