# Architecture Notes

## Data flow

FHIR JSON → ingestion → Bronze → Silver → Gold Patient 360 → SQL/Data Quality → RAG → AI Healthcare Insights.

Airflow is the orchestration layer. Structured Streaming is demonstrated separately in the end-to-end notebook.

## Design principles

- Preserve raw source fidelity in Bronze.
- Normalize clinical entities in Silver.
- Keep business-facing aggregates in Gold.
- Treat data quality as a gate before downstream AI/analytics consumption.
- Ground GenAI outputs in retrieved context.
- Keep orchestration separate from core Spark transformations.

## Operational considerations

Production deployment would additionally require secrets management, environment-specific configuration, monitoring/alerting, CI/CD, access controls, audit logging, and a deployed vector retrieval/model-serving layer if continuous GenAI serving is required.
