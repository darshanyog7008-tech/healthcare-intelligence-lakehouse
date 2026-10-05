# Healthcare Intelligence Lakehouse

An end-to-end healthcare data engineering and GenAI portfolio project built around FHIR/Synthea data, PySpark, Delta Lake, SQL analytics, data quality, RAG, healthcare insights, Structured Streaming, and Airflow orchestration.

> **Portfolio note:** This repository contains a Databricks implementation/prototype. Cloud-specific catalog, volume, credentials, and Databricks Job IDs are environment-dependent and are intentionally not committed.

## Architecture

```text
Synthea / FHIR JSON
        |
        v
+------------------+
| FHIR Ingestion   |
+------------------+
        |
        v
+------------------+
| Bronze / Raw     |
+------------------+
        |
        v
+------------------+
| Silver / Curated |
+------------------+
        |
        v
+------------------+
| Gold / Patient360|
+------------------+
        |
        +--------------------+
        |                    |
        v                    v
   SQL Analytics        Data Quality
                             |
                             v
                       RAG Corpus
                             |
                             v
                    AI Healthcare Insights
                             |
                             v
                   Airflow Orchestration

Structured Streaming is demonstrated in the end-to-end notebook.
```

## Notebook sequence

| Notebook | Purpose |
|---|---|
| `00_Project_Overview.py` | Architecture, goals, and project design |
| `01_FHIR_Ingestion.py` | FHIR source ingestion and parsing |
| `02_Bronze_Layer.py` | Raw Bronze Delta layer |
| `03_Silver_Layer.py` | Curated clinical entities |
| `04_Gold_Healthcare_Analytics.py` | Gold analytics and Patient 360 |
| `05_SQL_Analytics.py` | Healthcare analytical SQL |
| `06_Data_Quality.py` | Data-quality validation and gates |
| `07_GenAI_RAG.py` | RAG corpus and grounded retrieval workflow |
| `08_AI_Healthcare_Insights.py` | Patient-level healthcare insights |
| `09_End_to_End.py` | Final integration, streaming demo, Airflow design, and validation |

## Technology stack

- Databricks
- Apache Spark / PySpark
- Delta Lake
- Spark SQL
- FHIR / Synthea
- Apache Airflow
- Retrieval-Augmented Generation (RAG)
- GenAI / LLM prompting
- Structured Streaming
- Git / GitHub

## Key engineering concepts demonstrated

- Medallion architecture
- Schema-aware FHIR processing
- Patient 360 modeling
- Distributed PySpark transformations
- SQL analytics and aggregations
- Data-quality gates
- RAG document construction
- Grounded healthcare insight generation
- Streaming checkpointing
- Workflow orchestration
- End-to-end pipeline validation

## Running in Databricks

1. Import the notebooks under `notebooks/` into a Databricks workspace.
2. Configure your Unity Catalog catalog/schema/volume names for your environment.
3. Upload or connect the FHIR/Synthea source data.
4. Run notebooks in numeric order from `00` through `09`.
5. Configure an Airflow Databricks connection and Job ID before deploying `airflow/healthcare_pipeline_dag.py`.

The notebook sources currently reference the project's Databricks catalog/table names. If you use a different catalog, update those references consistently.

## RAG / GenAI scope

The RAG portion builds and validates a grounded corpus and retrieval workflow for healthcare information. It should be described in interviews as a **RAG prototype/baseline**, unless a production vector index and model-serving endpoint are separately deployed.

## Streaming scope

`09_End_to_End.py` contains a lightweight Structured Streaming demonstration with an explicit checkpoint location. It is intended to demonstrate the Spark streaming API rather than claim a continuously running production ingestion service.

## Airflow scope

The `airflow/` DAG provides the orchestration scaffold. The Databricks Job ID and Airflow connection are environment-specific and must be configured before deployment.

## Interview walkthrough

A concise interview narrative is:

1. Start with the FHIR source and ingestion strategy.
2. Explain why Bronze preserves source fidelity.
3. Show how Silver normalizes major clinical entities.
4. Explain Gold Patient 360 and analytical use cases.
5. Demonstrate SQL analytics and data-quality gates.
6. Explain how patient/knowledge documents are prepared for RAG.
7. Show how grounded AI insights are generated.
8. Explain Structured Streaming and checkpointing.
9. Explain Airflow's role as the orchestration layer.
10. Finish with validation, operational trade-offs, and GitHub packaging.

## Disclaimer

This is an engineering/analytics portfolio project using synthetic healthcare data. It is not a clinical decision-support system and should not be used for real patient care.
