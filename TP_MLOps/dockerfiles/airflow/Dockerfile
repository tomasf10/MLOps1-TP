FROM apache/airflow:2.10.4-python3.11

USER root

COPY pyproject.toml /opt/project/pyproject.toml
COPY src /opt/project/src

RUN chown -R airflow:0 /opt/project

USER airflow

RUN pip install --no-cache-dir "/opt/project"
