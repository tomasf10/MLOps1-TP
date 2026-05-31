from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from diabetes_mlops.config import settings


FEATURE_COLUMNS = [
    "gender",
    "age",
    "location",
    "race",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "hba1c_level",
    "blood_glucose_level",
]
TARGET_COLUMN = "diabetes"
ALT_TARGET_COLUMNS = ["diabetes_status", "diabetes_binary", "outcome"]


def normalize_column_name(column: str) -> str:
    clean = column.strip().lower()
    clean = re.sub(r"[^a-z0-9]+", "_", clean)
    return clean.strip("_")


def load_dataset(path: str | Path = settings.dataset_path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No encontramos el dataset en {dataset_path}. "
            "Dejalo en data/raw/diabetes_dataset.csv o seteá DATASET_PATH."
        )

    dataset = pd.read_csv(dataset_path)
    dataset.columns = [normalize_column_name(column) for column in dataset.columns]
    dataset = _adapt_target_name(dataset)
    dataset = _adapt_race_columns(dataset)
    dataset = _coerce_target(dataset)
    validate_dataset(dataset)
    return dataset[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()


def _adapt_target_name(dataset: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN in dataset.columns:
        return dataset

    for candidate in ALT_TARGET_COLUMNS:
        if candidate in dataset.columns:
            return dataset.rename(columns={candidate: TARGET_COLUMN})

    return dataset


def _adapt_race_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    if "race" in dataset.columns:
        return dataset

    race_columns = {
        "race_africanamerican": "African American",
        "race_asian": "Asian",
        "race_caucasian": "Caucasian",
        "race_hispanic": "Hispanic",
        "race_other": "Other",
    }
    present_columns = [column for column in race_columns if column in dataset.columns]
    if not present_columns:
        return dataset

    dataset = dataset.copy()

    def pick_race(row: pd.Series) -> str:
        for column in present_columns:
            if int(row[column]) == 1:
                return race_columns[column]
        return "Unknown"

    dataset["race"] = dataset[present_columns].apply(pick_race, axis=1)
    return dataset


def _coerce_target(dataset: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in dataset.columns:
        return dataset

    target = dataset[TARGET_COLUMN]
    if target.dtype == object:
        mapped = target.astype(str).str.strip().str.lower().map(
            {
                "1": 1,
                "0": 0,
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "diabetic": 1,
                "non_diabetic": 0,
                "non-diabetic": 0,
                "positive": 1,
                "negative": 0,
            }
        )
        if mapped.notna().all():
            dataset = dataset.copy()
            dataset[TARGET_COLUMN] = mapped.astype(int)

    return dataset


def validate_dataset(dataset: pd.DataFrame) -> None:
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in dataset]
    if missing:
        raise ValueError(
            "Faltan columnas esperadas en el CSV: "
            + ", ".join(missing)
            + ". Revisar nombres de columnas o adaptar el mapper."
        )

    if dataset.empty:
        raise ValueError("El dataset esta vacio.")

    target_values = set(dataset[TARGET_COLUMN].dropna().unique().tolist())
    if not target_values.issubset({0, 1, False, True}):
        raise ValueError("La columna diabetes tiene que ser binaria: 0/1 o true/false.")


def split_features_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_dataset(dataset)
    features = dataset[FEATURE_COLUMNS].copy()
    target = dataset[TARGET_COLUMN].astype(int).copy()
    return features, target


def dataset_summary(dataset: pd.DataFrame) -> dict[str, int | list[str]]:
    return {
        "rows": len(dataset),
        "columns": list(dataset.columns),
        "positive_cases": int(dataset[TARGET_COLUMN].sum()),
        "negative_cases": int((dataset[TARGET_COLUMN] == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida la carga del CSV de diabetes.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=settings.dataset_path,
        help="Ruta al CSV. Por defecto usa data/raw/diabetes_dataset.csv.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)
    summary = dataset_summary(dataset)
    print(f"CSV cargado correctamente: {summary['rows']} filas")
    print(f"Columnas: {', '.join(summary['columns'])}")
    print(
        "Target diabetes: "
        f"{summary['positive_cases']} positivos / {summary['negative_cases']} negativos"
    )


if __name__ == "__main__":
    main()
