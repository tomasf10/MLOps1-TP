# TP MLOps

## Status Actual

- Carga y validacion del CSV.
- Un pipeline de entrenamiento con scikit-learn.
- Un DAG de Airflow que ejecuta la validacion y el entrenamiento.
- MLflow para guardar metricas y artefactos.
- MinIO como storage local tipo S3 para los artefactos de MLflow.
- Postgres para metadata de Airflow y MLflow.

## Que hace cada parte

`data.py` carga el CSV, normaliza los nombres de columnas y valida que esten las columnas que necesita el modelo.

`features.py` arma el preprocesamiento:

- variables numericas con imputacion y escalado,
- variables categoricas con imputacion y one-hot encoding.

`model.py` arma un `RandomForestClassifier`.

`training.py` entrena el modelo, calcula metricas, guarda:

- `artifacts/model.joblib`
- `artifacts/metrics.json`

Tambien registra parametros, metricas y artefactos en MLflow.

`diabetes_training_pipeline.py` es el DAG de Airflow. Ejecuta:

1. validar el dataset,
2. entrenar el modelo,
3. mostrar un resumen del run.

## Correr con Docker

Pararse en la carpeta del proyecto:

cd TP_MLOps

docker compose down -v --remove-orphans
docker compose up --build airflow-init
docker compose up airflow-webserver airflow-scheduler mlflow

## URLs

Airflow: http://localhost:8080
MLflow: http://localhost:5001
MinIO: http://localhost:9001

## Correr el DAG

Entrar a Airflow, buscar el DAG. Activarlo y correrlo manualmente. Si termina bien, deberia aparecer:

- un run en MLflow,
- metricas como accuracy, precision, recall, f1 y roc_auc,
- el modelo en `artifacts/model.joblib`,
- las metricas en `artifacts/metrics.json`.

## Problemas que hubo

### `database "mlflow" does not exist` (toto)

Volumen viejo de Postgres creado antes de que exista la base `mlflow`.

Solucion:

docker compose down -v --remove-orphans
docker compose up --build airflow-init

### Falla por permisos sobre artifacts/model.joblib (tincho)

Se esta intentando acceder al model.joblib del local system. Deberiamos usar el S3.
Solucion simple para este TP:

bash
mkdir -p artifacts
chmod -R a+w artifacts
docker compose down -v --remove-orphans
docker compose up --build airflow-init

## Proximos pasos

* Agregar modelo a S3
* Conectar con FastAPI
* Mejorar modelo
* Agregar pipelines para monitoreo y reentramiento de modelos
