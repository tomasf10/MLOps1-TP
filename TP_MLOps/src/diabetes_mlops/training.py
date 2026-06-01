from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from diabetes_mlops.config import settings
from diabetes_mlops.data import TARGET_COLUMN, load_dataset, split_features_target
from diabetes_mlops.evaluation import evaluate_classifier
from diabetes_mlops.model import build_model


def train(
    dataset_path: Path = settings.dataset_path,
    model_output_path: Path = settings.model_output_path,
    metrics_output_path: Path = settings.metrics_output_path,
    tracking_uri: str = settings.mlflow_tracking_uri,
    experiment_name: str = settings.mlflow_experiment_name,
) -> dict[str, object]:
    dataset = load_dataset(dataset_path)
    X, y = split_features_target(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )

    model = build_model(random_state=settings.random_state)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="diabetes_random_forest") as run:
        model.fit(X_train, y_train)
        metrics = evaluate_classifier(model, X_test, y_test)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("target", TARGET_COLUMN)
        mlflow.log_param("rows", len(dataset))
        mlflow.log_param("test_size", settings.test_size)
        mlflow.log_param("random_state", settings.random_state)
        mlflow.log_params(model.named_steps["classifier"].get_params())

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_output_path)

        payload = {
            "run_id": run.info.run_id,
            "dataset_path": str(dataset_path),
            "model_path": str(model_output_path),
            "metrics": metrics,
        }
        metrics_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(model_output_path), artifact_path="local_artifacts")
        mlflow.log_artifact(str(metrics_output_path), artifact_path="local_artifacts")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modelo de diabetes del TP.")
    parser.add_argument("--dataset-path", type=Path, default=settings.dataset_path)
    parser.add_argument("--model-output-path", type=Path, default=settings.model_output_path)
    parser.add_argument("--metrics-output-path", type=Path, default=settings.metrics_output_path)
    parser.add_argument("--tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--experiment-name", default=settings.mlflow_experiment_name)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(
        dataset_path=args.dataset_path,
        model_output_path=args.model_output_path,
        metrics_output_path=args.metrics_output_path,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
