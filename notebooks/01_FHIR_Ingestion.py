# Databricks notebook source
# MAGIC %md
# MAGIC # FHIR Ingestion
# MAGIC
# MAGIC ## Notebook Objective
# MAGIC
# MAGIC This notebook is responsible for ingesting and inspecting the Synthea FHIR healthcare dataset.
# MAGIC
# MAGIC The source data is provided as FHIR JSON and contains multiple interconnected healthcare resources.
# MAGIC
# MAGIC The objectives of this notebook are to:
# MAGIC
# MAGIC - Understand the FHIR source structure
# MAGIC - Identify available healthcare resources
# MAGIC - Inspect the raw JSON structure
# MAGIC - Establish the source data contract
# MAGIC - Read the FHIR data using PySpark
# MAGIC - Inspect inferred and explicit schemas
# MAGIC - Validate the ingested data
# MAGIC - Capture basic ingestion statistics
# MAGIC - Define the strategy for the downstream Bronze layer
# MAGIC
# MAGIC This notebook focuses only on **source ingestion and inspection**.
# MAGIC
# MAGIC Data cleansing, normalization, business transformations, and analytical modelling will be performed in subsequent notebooks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Source Information
# MAGIC
# MAGIC ### Dataset
# MAGIC
# MAGIC **Synthea Synthetic Healthcare Data**
# MAGIC
# MAGIC ### Source Format
# MAGIC
# MAGIC **FHIR JSON**
# MAGIC
# MAGIC ### Data Type
# MAGIC
# MAGIC Synthetic healthcare / electronic health record data.
# MAGIC
# MAGIC ### Source Characteristics
# MAGIC
# MAGIC The dataset contains multiple FHIR resources representing different aspects of a synthetic patient's healthcare history.
# MAGIC
# MAGIC Examples include:
# MAGIC
# MAGIC - Patient
# MAGIC - Encounter
# MAGIC - Condition
# MAGIC - Observation
# MAGIC - Procedure
# MAGIC - Medication
# MAGIC - DiagnosticReport
# MAGIC - Immunization
# MAGIC - CarePlan
# MAGIC - CareTeam
# MAGIC - Claim
# MAGIC - ExplanationOfBenefit
# MAGIC - Practitioner
# MAGIC - Organization
# MAGIC
# MAGIC ### Ingestion Principle
# MAGIC
# MAGIC The source data will initially be treated as **raw source data**.
# MAGIC
# MAGIC No business transformations will be applied during the initial ingestion stage.
# MAGIC
# MAGIC The raw structure should be preserved so that downstream processing can reproduce or audit the source data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. FHIR Resource Overview
# MAGIC
# MAGIC FHIR (Fast Healthcare Interoperability Resources) is a healthcare data standard that represents healthcare information as interconnected resources.
# MAGIC
# MAGIC For this project, the important resources include:
# MAGIC
# MAGIC | Resource | Purpose |
# MAGIC |---|---|
# MAGIC | Patient | Synthetic patient demographic information |
# MAGIC | Encounter | Healthcare encounters or visits |
# MAGIC | Condition | Recorded health conditions |
# MAGIC | Observation | Clinical observations and measurements |
# MAGIC | Procedure | Procedures performed |
# MAGIC | Medication | Medication information |
# MAGIC | MedicationRequest | Medication requests/orders |
# MAGIC | MedicationAdministration | Medication administration records |
# MAGIC | DiagnosticReport | Diagnostic reports |
# MAGIC | Immunization | Immunization records |
# MAGIC | CarePlan | Patient care plans |
# MAGIC | CareTeam | Care team information |
# MAGIC | Claim | Healthcare claim information |
# MAGIC | ExplanationOfBenefit | Explanation of healthcare benefits |
# MAGIC | Practitioner | Healthcare practitioner information |
# MAGIC | Organization | Healthcare organization information |
# MAGIC | Device | Medical device information |
# MAGIC | ImagingStudy | Imaging study information |
# MAGIC | AllergyIntolerance | Allergy information |
# MAGIC | SupplyDelivery | Supply delivery information |
# MAGIC
# MAGIC The exact fields and relationships will be determined by inspecting the actual source files rather than assuming a fixed schema.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Ingestion Requirements
# MAGIC
# MAGIC The ingestion process should satisfy the following requirements:
# MAGIC
# MAGIC ### Source Preservation
# MAGIC
# MAGIC Raw source information should be preserved during ingestion.
# MAGIC
# MAGIC ### Schema Awareness
# MAGIC
# MAGIC The pipeline should understand the structure of the FHIR JSON rather than treating the data as an arbitrary JSON document.
# MAGIC
# MAGIC ### Repeatability
# MAGIC
# MAGIC The ingestion process should be repeatable without creating uncontrolled duplicate records.
# MAGIC
# MAGIC ### Validation
# MAGIC
# MAGIC Basic validation should be performed after ingestion, including:
# MAGIC
# MAGIC - File availability
# MAGIC - File readability
# MAGIC - Resource availability
# MAGIC - Record counts
# MAGIC - Schema inspection
# MAGIC - Required identifiers where applicable
# MAGIC
# MAGIC ### Observability
# MAGIC
# MAGIC The ingestion process should capture useful statistics such as:
# MAGIC
# MAGIC - Number of source files
# MAGIC - Number of resources
# MAGIC - Number of patients
# MAGIC - Resource types
# MAGIC - Record counts
# MAGIC - Schema information
# MAGIC
# MAGIC ### Separation of Responsibilities
# MAGIC
# MAGIC This notebook is responsible for ingestion and source inspection.
# MAGIC
# MAGIC The following activities belong to later notebooks:
# MAGIC
# MAGIC - Data cleansing
# MAGIC - Normalization
# MAGIC - Business transformations
# MAGIC - Gold-layer modelling
# MAGIC - Healthcare KPIs
# MAGIC - RAG

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Ingestion Flow
# MAGIC
# MAGIC The initial ingestion flow is:
# MAGIC
# MAGIC ```text
# MAGIC Synthea FHIR JSON
# MAGIC         |
# MAGIC         v
# MAGIC Source File Validation
# MAGIC         |
# MAGIC         v
# MAGIC FHIR JSON Inspection
# MAGIC         |
# MAGIC         v
# MAGIC PySpark Ingestion
# MAGIC         |
# MAGIC         v
# MAGIC Schema Inspection
# MAGIC         |
# MAGIC         v
# MAGIC Resource Validation
# MAGIC         |
# MAGIC         v
# MAGIC Ingestion Statistics
# MAGIC         |
# MAGIC         v
# MAGIC Bronze Layer

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Source Data Location
# MAGIC
# MAGIC The Synthea FHIR dataset is stored in a dedicated Unity Catalog managed volume.
# MAGIC
# MAGIC ### Source Configuration
# MAGIC
# MAGIC - Catalog: `healthcare_lakehouse`
# MAGIC - Schema: `default`
# MAGIC - Volume: `healthcare_source`
# MAGIC
# MAGIC The raw FHIR JSON files are stored under:
# MAGIC
# MAGIC `/Volumes/healthcare_lakehouse/default/healthcare_source/`
# MAGIC
# MAGIC The original source files are preserved without modification.
# MAGIC
# MAGIC The ingestion pipeline will read these files directly from the managed volume.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Source Inspection
# MAGIC
# MAGIC Before processing the healthcare data, the source files will be inspected to understand their structure.
# MAGIC
# MAGIC The inspection will identify:
# MAGIC
# MAGIC - Number of source files
# MAGIC - File names
# MAGIC - File sizes
# MAGIC - File formats
# MAGIC - FHIR resource types
# MAGIC - JSON structure
# MAGIC - Nested fields
# MAGIC - Resource identifiers
# MAGIC - Patient identifiers
# MAGIC - Available clinical information
# MAGIC
# MAGIC No analytical transformations will be performed at this stage.
# MAGIC
# MAGIC The purpose of this step is to understand the raw source before designing the Bronze layer.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Source Path Configuration
# MAGIC
# MAGIC The source path for the Synthea FHIR data is:
# MAGIC
# MAGIC `/Volumes/healthcare_lakehouse/default/healthcare_source/`
# MAGIC
# MAGIC This path will be used by the PySpark ingestion process.
# MAGIC
# MAGIC Keeping the source path explicitly defined allows the ingestion logic to remain organized and makes it easier to parameterize the pipeline later.

# COMMAND ----------

# Cell 9 — Inspect source files

source_path = "/Volumes/healthcare_lakehouse/default/healthcare_source/"

files = dbutils.fs.ls(source_path)

display(files)

# COMMAND ----------

# Cell 10 — Inspect source file names

for file_info in files:
    print(file_info.name)

# COMMAND ----------

# Cell 11 — Inspect FHIR resource structure

import json

patient_files = [
    f.path
    for f in files
    if f.name.endswith(".json")
    and "hospitalInformation" not in f.name
    and "practitionerInformation" not in f.name
]

print(f"Patient JSON files: {len(patient_files)}")

# Read one complete FHIR JSON document using Spark
sample_df = spark.read.option("multiline", "true").json(patient_files[0])

print("Sample file:")
print(patient_files[0])

print("\nSchema:")
sample_df.printSchema()

print("\nTop-level columns:")
print(sample_df.columns)

# COMMAND ----------

# Cell 12 — Inspect FHIR Bundle Structure

print("Top-level columns:")
print(sample_df.columns)

print("\nFHIR Resource Type:")
sample_df.select("resourceType").show(truncate=False)

print("\nBundle Type:")
sample_df.select("type").show(truncate=False)

print("\nNumber of entries:")
sample_df.selectExpr("size(entry) as number_of_resources").show()

# COMMAND ----------

# Cell 13 — Extract individual FHIR resources

resources_df = (
    sample_df
    .selectExpr("explode(entry) as entry")
    .select("entry.resource.*")
)

print("Extracted resource columns:")
print(resources_df.columns)

print("\nNumber of resources:")
print(resources_df.count())

# COMMAND ----------

# Cell 14 — Analyze FHIR resource types

resource_type_counts = (
    resources_df
    .groupBy("resourceType")
    .count()
    .orderBy("count", ascending=False)
)

display(resource_type_counts)

# COMMAND ----------

# Cell 15 — Define core clinical FHIR resources

core_resources = [
    "Patient",
    "Encounter",
    "Condition",
    "Observation",
    "Procedure",
    "DiagnosticReport",
    "MedicationRequest",
    "MedicationAdministration",
    "Immunization",
    "AllergyIntolerance",
    "CarePlan",
    "CareTeam",
    "ImagingStudy"
]

print("Core clinical resources:")
for resource in core_resources:
    print(f"- {resource}")

# COMMAND ----------

