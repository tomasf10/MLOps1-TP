from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from airflow.decorators import dag, task


default_args = {
    "owner": "equipo-tp-mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": dt.timedelta(minutes=2),
}


@dag(
    dag_id="diabetes_training_pipeline",
    description="Valida datos, entrena el modelo y registra el run en MLflow.",
    start_date=dt.datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["mlops", "diabetes", "training", "mlflow"],
)
def diabetes_training_pipeline():
    @task
    def validate_dataset() -> str:
        from diabetes_mlops.config import settings
        from diabetes_mlops.data import dataset_summary, load_dataset

        dataset = load_dataset(settings.dataset_path)
        summary = dataset_summary(dataset)
        print(f"CSV validado: {settings.dataset_path}")
        print(f"Filas: {summary['rows']}")
        print(f"Columnas: {', '.join(summary['columns'])}")
        print(
            "Target diabetes: "
            f"{summary['positive_cases']} positivos / "
            f"{summary['negative_cases']} negativos"
        )
        return str(settings.dataset_path)

    @task
    def train_model(dataset_path: str) -> dict:
        from diabetes_mlops.config import settings
        from diabetes_mlops.training import train

        result = train(
            dataset_path=Path(dataset_path),
            model_output_path=settings.model_output_path,
            metrics_output_path=settings.metrics_output_path,
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
        )
        print(json.dumps(result, indent=2))
        return result

    @task
    def summarize_run(result: dict) -> None:
        metrics = result.get("metrics", {})
        print("Run terminado. Metricas principales:")
        for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if name in metrics:
                print(f"- {name}: {metrics[name]}")
        print(f"Modelo local: {result.get('model_path')}")
        print(f"MLflow run_id: {result.get('run_id')}")

    summarize_run(train_model(validate_dataset()))


diabetes_training_pipeline()
