from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from diabetes_mlops.data import FEATURE_COLUMNS


CATEGORICAL_COLUMNS = ["gender", "location", "race", "smoking_history"]
NUMERIC_COLUMNS = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "hba1c_level",
    "blood_glucose_level",
]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def check_feature_lists_are_aligned() -> None:
    expected = sorted(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
    actual = sorted(FEATURE_COLUMNS)
    if expected != actual:
        raise RuntimeError("Las columnas del preprocessor no matchean con el contrato.")
