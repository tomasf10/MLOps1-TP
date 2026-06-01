from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from diabetes_mlops.features import build_preprocessor, check_feature_lists_are_aligned


def build_model(random_state: int = 42) -> Pipeline:
    check_feature_lists_are_aligned()
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=8,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
