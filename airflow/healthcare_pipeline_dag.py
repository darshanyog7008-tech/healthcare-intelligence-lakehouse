"""Airflow orchestration for the Healthcare Intelligence Lakehouse.

Environment-specific values are placeholders by design. Configure the
Databricks connection and existing cluster before deploying.
"""

from datetime import datetime

from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksSubmitRunOperator

DATABRICKS_CLUSTER_ID = "REPLACE_WITH_CLUSTER_ID"
DATABRICKS_NOTEBOOK_ROOT = "/Workspace/Repos/<user>/Healthcare-Intelligence-Lakehouse/notebooks"

NOTEBOOKS = [
    "01_FHIR_Ingestion",
    "02_Bronze_Layer",
    "03_Silver_Layer",
    "04_Gold_Healthcare_Analytics",
    "05_SQL_Analytics",
    "06_Data_Quality",
    "07_GenAI_RAG",
    "08_AI_Healthcare_Insights",
    "09_End_to_End",
]

with DAG(
    dag_id="healthcare_data_engineering_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["healthcare", "databricks", "pyspark", "rag", "genai"],
    description="End-to-end healthcare lakehouse orchestration",
) as dag:
    tasks = []

    for notebook_name in NOTEBOOKS:
        task = DatabricksSubmitRunOperator(
            task_id=notebook_name.lower(),
            databricks_conn_id="databricks_default",
            existing_cluster_id=DATABRICKS_CLUSTER_ID,
            notebook_task={
                "notebook_path": f"{DATABRICKS_NOTEBOOK_ROOT}/{notebook_name}",
            },
        )
        tasks.append(task)

    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
