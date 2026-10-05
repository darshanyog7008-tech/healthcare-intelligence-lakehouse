# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Silver Layer
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC The Silver layer transforms the raw Bronze FHIR resources into cleaner, structured, and analysis-ready healthcare datasets.
# MAGIC
# MAGIC The Silver layer will:
# MAGIC
# MAGIC - Read data from the Bronze Delta table
# MAGIC - Separate FHIR resources by resource type
# MAGIC - Standardize important fields
# MAGIC - Flatten commonly used nested structures
# MAGIC - Handle missing and inconsistent values
# MAGIC - Create clean datasets for downstream analytics
# MAGIC - Preserve relationships between patients and clinical events
# MAGIC
# MAGIC ### Source
# MAGIC
# MAGIC Bronze table:
# MAGIC
# MAGIC `healthcare_lakehouse.bronze.fhir_resources`
# MAGIC
# MAGIC ### Target
# MAGIC
# MAGIC Silver schema:
# MAGIC
# MAGIC `healthcare_lakehouse.silver`
# MAGIC
# MAGIC ### Initial Clinical Domains
# MAGIC
# MAGIC - Patient
# MAGIC - Encounter
# MAGIC - Condition
# MAGIC - Observation
# MAGIC - Procedure
# MAGIC - DiagnosticReport
# MAGIC - MedicationRequest
# MAGIC - MedicationAdministration
# MAGIC - Immunization
# MAGIC - AllergyIntolerance

# COMMAND ----------

# Cell 2 — Import required libraries

from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

# Cell 3 — Create Silver schema

spark.sql("""
CREATE SCHEMA IF NOT EXISTS healthcare_lakehouse.silver
""")

print("Silver schema is ready.")

# COMMAND ----------

# Cell 4 — Load Bronze data

bronze_df = spark.table(
    "healthcare_lakehouse.bronze.fhir_resources"
)

print(f"Bronze records loaded: {bronze_df.count():,}")

# COMMAND ----------

# Cell 5 — Inspect available resource types

(
    bronze_df
    .groupBy("resourceType")
    .count()
    .orderBy(F.desc("count"))
    .show(50, truncate=False)
)

# COMMAND ----------

# Cell 6 — Extract Patient resources

patient_silver_df = (
    bronze_df
    .filter(F.col("resourceType") == "Patient")
    .select(
        F.col("id").alias("patient_id"),
        F.col("birthDate").alias("birth_date"),
        F.col("gender").alias("gender"),
        F.col("maritalStatus").alias("marital_status"),
        F.col("multipleBirthBoolean").alias("multiple_birth"),
        F.col("name").alias("name"),
        F.col("address").alias("address"),
        F.col("telecom").alias("telecom")
    )
)

print(f"Patient records: {patient_silver_df.count():,}")

patient_silver_df.printSchema()

# COMMAND ----------

# Cell 7 — Patient data quality checks

patient_silver_df.select(
    F.count("*").alias("total_records"),
    F.countDistinct("patient_id").alias("distinct_patients"),
    F.sum(
        F.when(F.col("patient_id").isNull(), 1).otherwise(0)
    ).alias("null_patient_ids"),
    F.sum(
        F.when(F.col("birth_date").isNull(), 1).otherwise(0)
    ).alias("null_birth_dates"),
    F.sum(
        F.when(F.col("gender").isNull(), 1).otherwise(0)
    ).alias("null_gender")
).show()

# COMMAND ----------

# Cell 8 — Inspect patient demographics

patient_silver_df.select(
    "patient_id",
    "birth_date",
    "gender",
    "marital_status",
    "multiple_birth"
).show(10, truncate=False)

# COMMAND ----------

# Cell 9 — Clean and standardize Patient demographics

patient_clean_df = (
    patient_silver_df
    .select(
        "patient_id",
        "birth_date",
        F.lower(F.trim(F.col("gender"))).alias("gender"),
        F.col("marital_status.text").alias("marital_status"),
        F.col("multiple_birth").alias("multiple_birth")
    )
)

print(f"Patient records: {patient_clean_df.count():,}")

patient_clean_df.show(10, truncate=False)

# COMMAND ----------

# Cell 10 — Patient data quality validation

patient_clean_df.select(
    F.count("*").alias("total_records"),
    F.countDistinct("patient_id").alias("distinct_patient_ids"),
    F.sum(F.when(F.col("patient_id").isNull(), 1).otherwise(0)).alias("null_patient_ids"),
    F.sum(F.when(F.col("birth_date").isNull(), 1).otherwise(0)).alias("null_birth_dates"),
    F.sum(F.when(F.col("gender").isNull(), 1).otherwise(0)).alias("null_gender"),
    F.sum(F.when(F.col("marital_status").isNull(), 1).otherwise(0)).alias("null_marital_status")
).show()

# COMMAND ----------

# Cell 11 — Create Silver Patient table

(
    patient_clean_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable("healthcare_lakehouse.silver.patient")
)

print("Silver Patient table created successfully.")

# COMMAND ----------

# Cell 12 — Verify Silver Patient table

silver_patient_df = spark.table(
    "healthcare_lakehouse.silver.patient"
)

print(f"Silver Patient records: {silver_patient_df.count():,}")

silver_patient_df.show(10, truncate=False)

# COMMAND ----------

# Cell 13 — Extract Encounter resources

encounter_silver_df = (
    bronze_df
    .filter(F.col("resourceType") == "Encounter")
    .select(
        F.col("id").alias("encounter_id"),
        F.col("subject.reference").alias("patient_reference"),
        F.col("status").alias("status"),
        F.col("class.code").alias("encounter_class"),
        F.col("type").alias("encounter_type"),
        F.col("period.start").alias("start_time"),
        F.col("period.end").alias("end_time"),
        F.col("serviceProvider.reference").alias("service_provider")
    )
)

print(f"Encounter records: {encounter_silver_df.count():,}")

encounter_silver_df.printSchema()

# COMMAND ----------

# Cell 14 — Standardize patient reference

encounter_clean_df = (
    encounter_silver_df
    .withColumn(
        "patient_id",
        F.regexp_extract(
            F.col("patient_reference"),
            r"Patient/(.+)",
            1
        )
    )
    .drop("patient_reference")
)

encounter_clean_df.select(
    "encounter_id",
    "patient_id",
    "status",
    "encounter_class",
    "start_time",
    "end_time"
).show(10, truncate=False)

# COMMAND ----------

# Cell 15 — Encounter data quality

encounter_clean_df.select(
    F.count("*").alias("total_encounters"),
    F.countDistinct("encounter_id").alias("distinct_encounters"),
    F.sum(
        F.when(F.col("encounter_id").isNull(), 1).otherwise(0)
    ).alias("null_encounter_ids"),
    F.sum(
        F.when(F.col("patient_id") == "", 1).otherwise(0)
    ).alias("missing_patient_ids"),
    F.sum(
        F.when(F.col("start_time").isNull(), 1).otherwise(0)
    ).alias("null_start_times")
).show()

# COMMAND ----------

# Cell 16 — Create Silver Encounter table

(
    encounter_clean_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(
        "healthcare_lakehouse.silver.encounter"
    )
)

print("Silver Encounter table created successfully.")

# COMMAND ----------

# Cell 17 — Verify Silver Encounter

silver_encounter_df = spark.table(
    "healthcare_lakehouse.silver.encounter"
)

print(
    f"Silver Encounter records: "
    f"{silver_encounter_df.count():,}"
)

silver_encounter_df.show(10, truncate=False)

# COMMAND ----------

# Cell 18 — Inspect Condition schema

condition_raw_df = (
    bronze_df
    .filter(F.col("resourceType") == "Condition")
)

print("Condition schema:")
condition_raw_df.printSchema()

# COMMAND ----------

# Cell 19 — Inspect Condition code fields

condition_raw_df.select(
    "id",
    "code",
    "clinicalStatus",
    "verificationStatus",
    "subject",
    "encounter",
    "onsetDateTime",
    "recordedDate"
).show(5, truncate=False)

# COMMAND ----------

# Cell 18 — Extract Condition resources

condition_silver_df = (
    bronze_df
    .filter(F.col("resourceType") == "Condition")
    .select(
        F.col("id").alias("condition_id"),
        F.col("subject").alias("patient_reference"),
        F.col("encounter").alias("encounter_reference"),
        F.col("code").alias("condition_code_json"),
        F.col("clinicalStatus").alias("clinical_status_json"),
        F.col("verificationStatus").alias("verification_status_json"),
        F.col("onsetDateTime").alias("onset_date"),
        F.col("recordedDate").alias("recorded_date")
    )
)

print(f"Condition records: {condition_silver_df.count():,}")

condition_silver_df.printSchema()

# COMMAND ----------

# Cell 19 — Inspect Condition code JSON

condition_silver_df.select(
    "condition_id",
    "condition_code_json",
    "clinical_status_json",
    "verification_status_json"
).show(5, truncate=False)

# COMMAND ----------

# Cell 18 — Extract Condition resources

condition_silver_df = (
    bronze_df
    .filter(F.col("resourceType") == "Condition")
    .select(
        F.col("id").alias("condition_id"),
        F.col("subject.reference").alias("patient_reference"),
        F.col("encounter.reference").alias("encounter_reference"),

        # code is stored as JSON string
        F.col("code").alias("condition_code_json"),

        # nested FHIR structures
        F.col("clinicalStatus.coding")[0]["code"].alias("clinical_status"),
        F.col("verificationStatus.coding")[0]["code"].alias("verification_status"),

        F.col("onsetDateTime").alias("onset_date"),
        F.col("recordedDate").alias("recorded_date")
    )
)

print(f"Condition records: {condition_silver_df.count():,}")

# COMMAND ----------

# Cell 19 — Extract condition code and name from JSON string

condition_clean_df = (
    condition_silver_df
    .withColumn(
        "condition_code",
        F.regexp_extract(
            F.col("condition_code_json"),
            r'"code":"([^"]+)"',
            1
        )
    )
    .withColumn(
        "condition_name",
        F.regexp_extract(
            F.col("condition_code_json"),
            r'"display":"([^"]+)"',
            1
        )
    )
    .withColumn(
        "patient_id",
        F.regexp_extract(
            F.col("patient_reference"),
            r"Patient/(.+)",
            1
        )
    )
    .withColumn(
        "encounter_id",
        F.regexp_extract(
            F.col("encounter_reference"),
            r"Encounter/(.+)",
            1
        )
    )
    .drop(
        "patient_reference",
        "encounter_reference",
        "condition_code_json"
    )
)

condition_clean_df.select(
    "condition_id",
    "patient_id",
    "encounter_id",
    "condition_code",
    "condition_name",
    "clinical_status",
    "verification_status",
    "onset_date"
).show(10, truncate=False)

# COMMAND ----------

# Cell 20 — Condition data quality

condition_clean_df.select(
    F.count("*").alias("total_conditions"),
    F.countDistinct("condition_id").alias("distinct_condition_ids"),

    F.sum(
        F.when(
            F.col("patient_id").isNull() |
            (F.col("patient_id") == ""),
            1
        ).otherwise(0)
    ).alias("missing_patient_ids"),

    F.sum(
        F.when(
            F.col("condition_name").isNull() |
            (F.col("condition_name") == ""),
            1
        ).otherwise(0)
    ).alias("missing_condition_names"),

    F.sum(
        F.when(
            F.col("condition_code").isNull() |
            (F.col("condition_code") == ""),
            1
        ).otherwise(0)
    ).alias("missing_condition_codes")
).show()

# COMMAND ----------

# Cell 21 — Create Silver Condition table

(
    condition_clean_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(
        "healthcare_lakehouse.silver.condition"
    )
)

print("Silver Condition table created successfully.")

# COMMAND ----------

# Cell 22 — Verify Silver Condition

silver_condition_df = spark.table(
    "healthcare_lakehouse.silver.condition"
)

print(
    f"Silver Condition records: "
    f"{silver_condition_df.count():,}"
)

silver_condition_df.show(10, truncate=False)

# COMMAND ----------

# Cell 23 — Extract Observation resources

observation_raw_df = (
    bronze_df
    .filter(F.col("resourceType") == "Observation")
)

print(f"Observation records: {observation_raw_df.count():,}")

observation_raw_df.printSchema()

# COMMAND ----------

# Cell 24 — Inspect Observation fields

observation_raw_df.select(
    "id",
    "status",
    "code",
    "subject",
    "encounter",
    "effectiveDateTime",
    "valueQuantity",
    "valueCodeableConcept"
).show(5, truncate=False)

# COMMAND ----------

# Cell 25 — Extract and clean Observation resources

observation_clean_df = (
    observation_raw_df
    .select(
        F.col("id").alias("observation_id"),
        F.col("status").alias("status"),

        # Observation code is stored as JSON string
        F.regexp_extract(
            F.col("code"),
            r'"code":"([^"]+)"',
            1
        ).alias("observation_code"),

        F.regexp_extract(
            F.col("code"),
            r'"display":"([^"]+)"',
            1
        ).alias("observation_name"),

        # Patient / Encounter references
        F.regexp_extract(
            F.col("subject.reference"),
            r"Patient/(.+)",
            1
        ).alias("patient_id"),

        F.regexp_extract(
            F.col("encounter.reference"),
            r"Encounter/(.+)",
            1
        ).alias("encounter_id"),

        F.col("effectiveDateTime").alias("observation_time"),

        # Quantity result
        F.col("valueQuantity.value").alias("value"),

        F.col("valueQuantity.unit").alias("unit"),

        F.col("valueQuantity.system").alias("unit_system")
    )
)

print(
    f"Silver Observation records: "
    f"{observation_clean_df.count():,}"
)

observation_clean_df.printSchema()

# COMMAND ----------

# Cell 26 — Inspect cleaned observations

observation_clean_df.select(
    "observation_id",
    "patient_id",
    "observation_code",
    "observation_name",
    "observation_time",
    "value",
    "unit"
).show(15, truncate=False)

# COMMAND ----------

# Cell 27 — Observation data quality

observation_clean_df.select(
    F.count("*").alias("total_observations"),

    F.countDistinct("observation_id")
        .alias("distinct_observation_ids"),

    F.sum(
        F.when(
            F.col("patient_id").isNull() |
            (F.col("patient_id") == ""),
            1
        ).otherwise(0)
    ).alias("missing_patient_ids"),

    F.sum(
        F.when(
            F.col("observation_code").isNull() |
            (F.col("observation_code") == ""),
            1
        ).otherwise(0)
    ).alias("missing_observation_codes"),

    F.sum(
        F.when(
            F.col("observation_name").isNull() |
            (F.col("observation_name") == ""),
            1
        ).otherwise(0)
    ).alias("missing_observation_names")
).show()

# COMMAND ----------

# Cell 28 — Most common observation types

(
    observation_clean_df
    .groupBy(
        "observation_code",
        "observation_name",
        "unit"
    )
    .count()
    .orderBy(F.desc("count"))
    .show(20, truncate=False)
)

# COMMAND ----------

# Cell 29 — Create Silver Observation table

(
    observation_clean_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(
        "healthcare_lakehouse.silver.observation"
    )
)

print("Silver Observation table created successfully.")

# COMMAND ----------

# Cell 30 — Verify Silver Observation

silver_observation_df = spark.table(
    "healthcare_lakehouse.silver.observation"
)

print(
    f"Silver Observation records: "
    f"{silver_observation_df.count():,}"
)

silver_observation_df.show(10, truncate=False)

# COMMAND ----------

# Cell 31 — Extract Procedure resources

procedure_raw_df = (
    bronze_df
    .filter(F.col("resourceType") == "Procedure")
)

print(f"Procedure records: {procedure_raw_df.count():,}")

procedure_raw_df.printSchema()

# COMMAND ----------

# Cell 32 — Inspect Procedure structure

procedure_raw_df.select(
    "id",
    "status",
    "code",
    "subject",
    "encounter",
    "performedPeriod"
).show(5, truncate=False)

# COMMAND ----------

# Cell 33 — Extract and clean Procedure resources

procedure_clean_df = (
    procedure_raw_df
    .select(
        F.col("id").alias("procedure_id"),
        F.col("status").alias("status"),

        # Procedure code and name
        F.regexp_extract(
            F.col("code"),
            r'"code":"([^"]+)"',
            1
        ).alias("procedure_code"),

        F.regexp_extract(
            F.col("code"),
            r'"display":"([^"]+)"',
            1
        ).alias("procedure_name"),

        # Patient
        F.regexp_extract(
            F.col("subject.reference"),
            r"Patient/(.+)",
            1
        ).alias("patient_id"),

        # Encounter
        F.regexp_extract(
            F.col("encounter.reference"),
            r"Encounter/(.+)",
            1
        ).alias("encounter_id"),

        # Procedure period
        F.col("performedPeriod.start").alias("procedure_start"),
        F.col("performedPeriod.end").alias("procedure_end")
    )
)

print(
    f"Silver Procedure records: "
    f"{procedure_clean_df.count():,}"
)

procedure_clean_df.printSchema()

# COMMAND ----------

# Cell 34 — Inspect cleaned procedures

procedure_clean_df.select(
    "procedure_id",
    "patient_id",
    "encounter_id",
    "status",
    "procedure_code",
    "procedure_name",
    "procedure_start",
    "procedure_end"
).show(10, truncate=False)

# COMMAND ----------

# Cell 35 — Procedure data quality

procedure_clean_df.select(
    F.count("*").alias("total_procedures"),

    F.countDistinct("procedure_id")
        .alias("distinct_procedure_ids"),

    F.sum(
        F.when(
            F.col("patient_id").isNull() |
            (F.col("patient_id") == ""),
            1
        ).otherwise(0)
    ).alias("missing_patient_ids"),

    F.sum(
        F.when(
            F.col("procedure_name").isNull() |
            (F.col("procedure_name") == ""),
            1
        ).otherwise(0)
    ).alias("missing_procedure_names")
).show()

# COMMAND ----------

# Cell 36 — Most common procedures

(
    procedure_clean_df
    .groupBy(
        "procedure_code",
        "procedure_name"
    )
    .count()
    .orderBy(F.desc("count"))
    .show(20, truncate=False)
)

# COMMAND ----------

# Cell 36 — Most common procedures

(
    procedure_clean_df
    .groupBy(
        "procedure_code",
        "procedure_name"
    )
    .count()
    .orderBy(F.desc("count"))
    .show(20, truncate=False)
)

# COMMAND ----------

# Cell 37 — Create Silver Procedure table

(
    procedure_clean_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(
        "healthcare_lakehouse.silver.procedure"
    )
)

print("Silver Procedure table created successfully.")

# COMMAND ----------

# Cell 38 — Verify Silver Procedure

silver_procedure_df = spark.table(
    "healthcare_lakehouse.silver.procedure"
)

print(
    f"Silver Procedure records: "
    f"{silver_procedure_df.count():,}"
)

# COMMAND ----------

# Cell 39 — Extract MedicationRequest resources

med_request_raw_df = (
    bronze_df
    .filter(F.col("resourceType") == "MedicationRequest")
)

print(
    f"MedicationRequest records: "
    f"{med_request_raw_df.count():,}"
)

med_request_raw_df.printSchema()

# COMMAND ----------

# Cell 40 — Inspect MedicationRequest structure

med_request_raw_df.select(
    "id",
    "status",
    "intent",
    "medicationCodeableConcept",
    "medicationReference",
    "subject",
    "encounter",
    "authoredOn",
    "dosageInstruction"
).show(5, truncate=False)

# COMMAND ----------

# Cell 41 — Extract and clean MedicationRequest resources

medication_request_df = (
    med_request_raw_df
    .select(
        F.col("id").alias("medication_request_id"),
        F.col("status").alias("status"),
        F.col("intent").alias("intent"),

        # MedicationCodeableConcept
        F.col(
            "medicationCodeableConcept.coding"
        )[0]["code"].alias("medication_code"),

        F.col(
            "medicationCodeableConcept.coding"
        )[0]["display"].alias("medication_name"),

        # MedicationReference
        F.col(
            "medicationReference.reference"
        ).alias("medication_reference"),

        # Patient
        F.regexp_extract(
            F.col("subject.reference"),
            r"Patient/(.+)",
            1
        ).alias("patient_id"),

        # Encounter
        F.regexp_extract(
            F.col("encounter.reference"),
            r"Encounter/(.+)",
            1
        ).alias("encounter_id"),

        F.col("authoredOn").alias("authored_on")
    )
)

print(
    f"MedicationRequest records: "
    f"{medication_request_df.count():,}"
)

medication_request_df.printSchema()

# COMMAND ----------

# Cell 42 — Standardize medication representation

medication_request_clean_df = (
    medication_request_df
    .withColumn(
        "medication_type",
        F.when(
            F.col("medication_code").isNotNull(),
            F.lit("CodeableConcept")
        )
        .when(
            F.col("medication_reference").isNotNull(),
            F.lit("Reference")
        )
        .otherwise(
            F.lit("Unknown")
        )
    )
)

medication_request_clean_df.select(
    "medication_request_id",
    "patient_id",
    "medication_code",
    "medication_name",
    "medication_reference",
    "medication_type",
    "status",
    "authored_on"
).show(10, truncate=False)

# COMMAND ----------

# Cell 43 — MedicationRequest data quality

medication_request_clean_df.select(
    F.count("*").alias("total_medication_requests"),

    F.countDistinct("medication_request_id")
        .alias("distinct_medication_request_ids"),

    F.sum(
        F.when(
            F.col("patient_id").isNull() |
            (F.col("patient_id") == ""),
            1
        ).otherwise(0)
    ).alias("missing_patient_ids"),

    F.sum(
        F.when(
            F.col("medication_name").isNull() &
            F.col("medication_reference").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_medication_information")
).show()

# COMMAND ----------

# Cell 44 — Most common medications

(
    medication_request_clean_df
    .filter(F.col("medication_name").isNotNull())
    .groupBy(
        "medication_code",
        "medication_name"
    )
    .count()
    .orderBy(F.desc("count"))
    .show(20, truncate=False)
)

# COMMAND ----------

# Cell 45 — Create Silver MedicationRequest table

(
    medication_request_clean_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(
        "healthcare_lakehouse.silver.medication_request"
    )
)

print(
    "Silver MedicationRequest table created successfully."
)

# COMMAND ----------

# Cell 46 — Verify Silver MedicationRequest

silver_medication_request_df = spark.table(
    "healthcare_lakehouse.silver.medication_request"
)

print(
    f"Silver MedicationRequest records: "
    f"{silver_medication_request_df.count():,}"
)

# COMMAND ----------

# Cell 47 — Extract DiagnosticReport resources

diagnostic_report_raw_df = (
    bronze_df
    .filter(F.col("resourceType") == "DiagnosticReport")
)

print(
    f"DiagnosticReport records: "
    f"{diagnostic_report_raw_df.count():,}"
)

diagnostic_report_raw_df.printSchema()

# COMMAND ----------

# Cell 48 — Inspect DiagnosticReport structure

diagnostic_report_raw_df.select(
    "id",
    "status",
    "code",
    "subject",
    "encounter",
    "effectiveDateTime",
    "issued",
    "result"
).show(5, truncate=False)

# COMMAND ----------

# Cell 49 — Extract and clean DiagnosticReport resources

diagnostic_report_df = (
    diagnostic_report_raw_df
    .select(
        F.col("id").alias("diagnostic_report_id"),
        F.col("status").alias("status"),

        # Diagnostic report code is stored as JSON string
        F.regexp_extract(
            F.col("code"),
            r'"code":"([^"]+)"',
            1
        ).alias("diagnostic_code"),

        F.regexp_extract(
            F.col("code"),
            r'"display":"([^"]+)"',
            1
        ).alias("diagnostic_name"),

        # Patient
        F.regexp_extract(
            F.col("subject.reference"),
            r"Patient/(.+)",
            1
        ).alias("patient_id"),

        # Encounter
        F.regexp_extract(
            F.col("encounter.reference"),
            r"Encounter/(.+)",
            1
        ).alias("encounter_id"),

        F.col("effectiveDateTime").alias("effective_time"),

        F.col("issued").alias("issued_time")
    )
)

print(
    f"Silver DiagnosticReport records: "
    f"{diagnostic_report_df.count():,}"
)

diagnostic_report_df.printSchema()

# COMMAND ----------

# Cell 50 — Inspect cleaned DiagnosticReports

diagnostic_report_df.select(
    "diagnostic_report_id",
    "patient_id",
    "encounter_id",
    "status",
    "diagnostic_code",
    "diagnostic_name",
    "effective_time",
    "issued_time"
).show(10, truncate=False)

# COMMAND ----------

# Cell 51 — DiagnosticReport data quality

diagnostic_report_df.select(
    F.count("*").alias("total_reports"),

    F.countDistinct("diagnostic_report_id")
        .alias("distinct_report_ids"),

    F.sum(
        F.when(
            F.col("patient_id").isNull() |
            (F.col("patient_id") == ""),
            1
        ).otherwise(0)
    ).alias("missing_patient_ids"),

    F.sum(
        F.when(
            F.col("diagnostic_code").isNull() |
            (F.col("diagnostic_code") == ""),
            1
        ).otherwise(0)
    ).alias("missing_diagnostic_codes")
).show()

# COMMAND ----------

# Cell 52 — Most common diagnostic reports

(
    diagnostic_report_df
    .groupBy(
        "diagnostic_code",
        "diagnostic_name"
    )
    .count()
    .orderBy(F.desc("count"))
    .show(20, truncate=False)
)

# COMMAND ----------

# Cell 53 — Create Silver DiagnosticReport table

(
    diagnostic_report_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(
        "healthcare_lakehouse.silver.diagnostic_report"
    )
)

print(
    "Silver DiagnosticReport table created successfully."
)

# COMMAND ----------

# Cell 54 — Verify Silver DiagnosticReport

silver_diagnostic_report_df = spark.table(
    "healthcare_lakehouse.silver.diagnostic_report"
)

print(
    f"Silver DiagnosticReport records: "
    f"{silver_diagnostic_report_df.count():,}"
)

# COMMAND ----------

# Cell 55 — Extract Immunization resources

immunization_raw_df = (
    bronze_df
    .filter(F.col("resourceType") == "Immunization")
)

print(
    f"Immunization records: {immunization_raw_df.count():,}"
)

immunization_raw_df.printSchema()

# COMMAND ----------

# Cell 56 — Inspect Immunization structure

immunization_raw_df.select(
    "id",
    "status",
    "vaccineCode",
    "patient",
    "encounter",
    "occurrenceDateTime"
).show(5, truncate=False)

# COMMAND ----------

# Cell 57 — Extract and clean Immunization resources

immunization_clean_df = (
    immunization_raw_df
    .select(
        F.col("id").alias("immunization_id"),
        F.col("status").alias("status"),

        # Vaccine coding
        F.col("vaccineCode.coding")[0]["code"]
            .alias("vaccine_code"),

        F.col("vaccineCode.coding")[0]["display"]
            .alias("vaccine_name"),

        # Patient
        F.regexp_extract(
            F.col("patient.reference"),
            r"Patient/(.+)",
            1
        ).alias("patient_id"),

        # Encounter
        F.regexp_extract(
            F.col("encounter.reference"),
            r"Encounter/(.+)",
            1
        ).alias("encounter_id"),

        F.col("occurrenceDateTime")
            .alias("vaccination_time")
    )
)

print(
    f"Silver Immunization records: "
    f"{immunization_clean_df.count():,}"
)

immunization_clean_df.printSchema()

# COMMAND ----------

# Cell 58 — Inspect cleaned Immunization records

immunization_clean_df.select(
    "immunization_id",
    "patient_id",
    "encounter_id",
    "status",
    "vaccine_code",
    "vaccine_name",
    "vaccination_time"
).show(10, truncate=False)

# COMMAND ----------

# Cell 59 — Immunization data quality

immunization_clean_df.select(
    F.count("*").alias("total_immunizations"),

    F.countDistinct("immunization_id")
        .alias("distinct_immunization_ids"),

    F.sum(
        F.when(
            F.col("patient_id").isNull() |
            (F.col("patient_id") == ""),
            1
        ).otherwise(0)
    ).alias("missing_patient_ids"),

    F.sum(
        F.when(
            F.col("vaccine_code").isNull() |
            (F.col("vaccine_code") == ""),
            1
        ).otherwise(0)
    ).alias("missing_vaccine_codes")
).show()

# COMMAND ----------

# Cell 60 — Vaccine distribution

(
    immunization_clean_df
    .groupBy(
        "vaccine_code",
        "vaccine_name"
    )
    .count()
    .orderBy(F.desc("count"))
    .show(20, truncate=False)
)

# COMMAND ----------

# Cell 61 — Create Silver Immunization table

(
    immunization_clean_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(
        "healthcare_lakehouse.silver.immunization"
    )
)

print("Silver Immunization table created successfully.")

# COMMAND ----------

# Cell 62 — Verify Silver Immunization

silver_immunization_df = spark.table(
    "healthcare_lakehouse.silver.immunization"
)

print(
    f"Silver Immunization records: "
    f"{silver_immunization_df.count():,}"
)

# COMMAND ----------

# Cell 63 — Silver table inventory

silver_tables = [
    "patient",
    "encounter",
    "condition",
    "observation",
    "procedure",
    "medication_request",
    "diagnostic_report",
    "immunization"
]

print("SILVER LAYER TABLES")
print("=" * 50)

for table in silver_tables:
    df = spark.table(
        f"healthcare_lakehouse.silver.{table}"
    )

    print(
        f"{table:<25} {df.count():>10,} records"
    )

# COMMAND ----------

# Cell 64 — Duplicate primary-key validation

key_columns = {
    "patient": "patient_id",
    "encounter": "encounter_id",
    "condition": "condition_id",
    "observation": "observation_id",
    "procedure": "procedure_id",
    "medication_request": "medication_request_id",
    "diagnostic_report": "diagnostic_report_id",
    "immunization": "immunization_id"
}

print("DUPLICATE ID CHECK")
print("=" * 50)

for table, key in key_columns.items():

    df = spark.table(
        f"healthcare_lakehouse.silver.{table}"
    )

    duplicate_count = (
        df.groupBy(key)
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    print(
        f"{table:<25} duplicate IDs: {duplicate_count}"
    )

# COMMAND ----------

# Cell 65 — Patient reference integrity

silver_patient_df = spark.table(
    "healthcare_lakehouse.silver.patient"
)

patient_ids = (
    silver_patient_df
    .select("patient_id")
    .distinct()
)

clinical_tables = [
    "encounter",
    "condition",
    "observation",
    "procedure",
    "medication_request",
    "diagnostic_report",
    "immunization"
]

print("PATIENT REFERENCE CHECK")
print("=" * 50)

for table in clinical_tables:

    df = spark.table(
        f"healthcare_lakehouse.silver.{table}"
    )

    orphan_count = (
        df
        .filter(F.col("patient_id").isNotNull())
        .join(
            patient_ids,
            "patient_id",
            "left_anti"
        )
        .count()
    )

    print(
        f"{table:<25} orphan patients: {orphan_count}"
    )

# COMMAND ----------

# Cell 66 — Inspect actual encounter reference formats

for table in [
    "condition",
    "observation",
    "procedure",
    "medication_request",
    "diagnostic_report",
    "immunization"
]:

    df = spark.table(
        f"healthcare_lakehouse.silver.{table}"
    )

    print(f"\n===== {table} =====")

    df.select("encounter_id") \
      .filter(F.col("encounter_id").isNotNull()) \
      .show(3, truncate=False)

# COMMAND ----------

# Cell 67 — Fix encounter IDs from Bronze

silver_tables_with_encounter = {
    "condition": "Condition",
    "observation": "Observation",
    "procedure": "Procedure",
    "medication_request": "MedicationRequest",
    "diagnostic_report": "DiagnosticReport",
    "immunization": "Immunization"
}

for table, resource_type in silver_tables_with_encounter.items():

    silver_df = spark.table(
        f"healthcare_lakehouse.silver.{table}"
    )

    source_df = (
        bronze_df
        .filter(F.col("resourceType") == resource_type)
        .select(
            F.col("id").alias("_resource_id"),
            F.col("encounter.reference").alias("_encounter_ref")
        )
    )

    # Identify the primary ID column
    id_column = {
        "condition": "condition_id",
        "observation": "observation_id",
        "procedure": "procedure_id",
        "medication_request": "medication_request_id",
        "diagnostic_report": "diagnostic_report_id",
        "immunization": "immunization_id"
    }[table]

    fixed_df = (
        silver_df
        .drop("encounter_id")
        .join(
            source_df,
            F.col(id_column) == F.col("_resource_id"),
            "left"
        )
        .withColumn(
            "encounter_id",
            F.regexp_extract(
                F.col("_encounter_ref"),
                r"urn:uuid:([a-fA-F0-9-]+)",
                1
            )
        )
        .drop("_resource_id", "_encounter_ref")
    )

    (
        fixed_df
        .write
        .mode("overwrite")
        .format("delta")
        .saveAsTable(
            f"healthcare_lakehouse.silver.{table}"
        )
    )

    print(f"Fixed: {table}")

# COMMAND ----------

# Cell 68 — Verify encounter IDs

for table in [
    "condition",
    "observation",
    "procedure",
    "medication_request",
    "diagnostic_report",
    "immunization"
]:

    df = spark.table(
        f"healthcare_lakehouse.silver.{table}"
    )

    print(f"\n===== {table} =====")

    df.select("encounter_id") \
      .filter(
          F.col("encounter_id").isNotNull() &
          (F.col("encounter_id") != "")
      ) \
      .show(3, truncate=False)

# COMMAND ----------

# Cell 69 — Validate encounter references

silver_encounter_df = spark.table(
    "healthcare_lakehouse.silver.encounter"
)

encounter_ids = (
    silver_encounter_df
    .select("encounter_id")
    .distinct()
)

encounter_tables = [
    "condition",
    "observation",
    "procedure",
    "medication_request",
    "diagnostic_report",
    "immunization"
]

print("ENCOUNTER REFERENCE CHECK")
print("=" * 55)

for table in encounter_tables:

    df = spark.table(
        f"healthcare_lakehouse.silver.{table}"
    )

    orphan_count = (
        df
        .filter(
            F.col("encounter_id").isNotNull() &
            (F.col("encounter_id") != "")
        )
        .join(
            encounter_ids,
            "encounter_id",
            "left_anti"
        )
        .count()
    )

    print(
        f"{table:<25} orphan encounters: {orphan_count}"
    )

# COMMAND ----------

# Cell 70 — Final Silver record counts

silver_tables = {
    "patient": "patient_id",
    "encounter": "encounter_id",
    "condition": "condition_id",
    "observation": "observation_id",
    "procedure": "procedure_id",
    "medication_request": "medication_request_id",
    "diagnostic_report": "diagnostic_report_id",
    "immunization": "immunization_id"
}

print("FINAL SILVER LAYER VALIDATION")
print("=" * 60)

for table, key in silver_tables.items():

    df = spark.table(
        f"healthcare_lakehouse.silver.{table}"
    )

    total = df.count()
    distinct_ids = df.select(key).distinct().count()

    print(
        f"{table:<25} "
        f"records: {total:>8,} | "
        f"distinct IDs: {distinct_ids:>8,}"
    )

# COMMAND ----------

# Cell 71 — Primary key validation

print("PRIMARY KEY NULL CHECK")
print("=" * 60)

for table, key in silver_tables.items():

    df = spark.table(
        f"healthcare_lakehouse.silver.{table}"
    )

    null_count = (
        df.filter(
            F.col(key).isNull() |
            (F.col(key) == "")
        ).count()
    )

    print(
        f"{table:<25} null/empty {key}: {null_count}"
    )

# COMMAND ----------

# Cell 72 — Silver readiness check

print("=" * 60)
print("SILVER LAYER STATUS")
print("=" * 60)

print("✓ Patient normalized")
print("✓ Encounter normalized")
print("✓ Condition normalized")
print("✓ Observation normalized")
print("✓ Procedure normalized")
print("✓ MedicationRequest normalized")
print("✓ DiagnosticReport normalized")
print("✓ Immunization normalized")
print("✓ Encounter references validated")
print("✓ Primary keys validated")
print()
print("SILVER LAYER: READY FOR GOLD")

# COMMAND ----------

