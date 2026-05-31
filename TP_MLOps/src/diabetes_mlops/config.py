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


settings = Settings()
