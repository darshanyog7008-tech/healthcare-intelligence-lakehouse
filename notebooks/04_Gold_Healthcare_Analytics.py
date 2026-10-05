# Databricks notebook source
# Cell 1 — Gold Layer Setup

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Catalog and schemas
CATALOG = "healthcare_lakehouse"
SILVER = f"{CATALOG}.silver"
GOLD = f"{CATALOG}.gold"

print("Gold Layer Setup")
print("=" * 50)
print(f"Catalog : {CATALOG}")
print(f"Source  : {SILVER}")
print(f"Target  : {GOLD}")

# COMMAND ----------

# Cell 2 — Load Silver tables

patient_df = spark.table(f"{SILVER}.patient")
encounter_df = spark.table(f"{SILVER}.encounter")
condition_df = spark.table(f"{SILVER}.condition")
observation_df = spark.table(f"{SILVER}.observation")
procedure_df = spark.table(f"{SILVER}.procedure")
medication_request_df = spark.table(f"{SILVER}.medication_request")
diagnostic_report_df = spark.table(f"{SILVER}.diagnostic_report")
immunization_df = spark.table(f"{SILVER}.immunization")

print("Silver tables loaded successfully.")

# COMMAND ----------

# Cell 3 — Verify Gold schema

spark.sql(f"""
CREATE SCHEMA IF NOT EXISTS {GOLD}
""")

print(f"Gold schema ready: {GOLD}")

# COMMAND ----------

# Cell 4 — Patient 360 Foundation

patient_360_df = (
    patient_df
    .select(
        F.col("patient_id"),
        F.to_date("birth_date").alias("birth_date"),
        F.col("gender"),
        F.col("marital_status"),
        F.col("multiple_birth")
    )
    .withColumn(
        "birth_year",
        F.year("birth_date")
    )
)

print("Patient 360 foundation created")
print(f"Records: {patient_360_df.count():,}")

patient_360_df.show(10, truncate=False)

# COMMAND ----------

# Cell 5 — Derive Patient Age

patient_360_df = (
    patient_360_df
    .withColumn(
        "age",
        F.floor(
            F.months_between(
                F.current_date(),
                F.col("birth_date")
            ) / 12
        ).cast("int")
    )
)

patient_360_df.select(
    "patient_id",
    "birth_date",
    "gender",
    "age"
).show(10, truncate=False)

# COMMAND ----------

# Cell 6 — Patient Encounter Summary

encounter_summary_df = (
    encounter_df
    .groupBy("patient_id")
    .agg(
        F.countDistinct("encounter_id").alias("total_encounters"),
        F.min("start_time").alias("first_encounter_time"),
        F.max("start_time").alias("last_encounter_time")
    )
)

print("Encounter summary created")
print(f"Patients with encounters: {encounter_summary_df.count():,}")

encounter_summary_df.show(10, truncate=False)

# COMMAND ----------

# Check actual patient_id values in Silver Encounter

encounter_df.select(
    "encounter_id",
    "patient_id",
    "start_time"
).show(10, truncate=False)

# COMMAND ----------

# Check patient_id quality

encounter_df.select(
    F.count("*").alias("total_records"),
    F.count(
        F.when(
            F.col("patient_id").isNull() |
            (F.col("patient_id") == ""),
            True
        )
    ).alias("null_or_empty_patient_id"),
    F.countDistinct("patient_id").alias("distinct_patient_ids")
).show()

# COMMAND ----------

# Cell 7 — Fix Encounter → Patient Reference

# Load Bronze Encounter resources directly
bronze_encounter_df = (
    spark.table("healthcare_lakehouse.bronze.fhir_resources")
    .filter(F.col("resourceType") == "Encounter")
)

encounter_source_df = (
    bronze_encounter_df
    .select(
        F.col("id").alias("_encounter_id"),
        F.col("subject.reference").alias("_patient_ref")
    )
)

fixed_encounter_df = (
    encounter_df
    .drop("patient_id")
    .join(
        encounter_source_df,
        encounter_df["encounter_id"] == F.col("_encounter_id"),
        "left"
    )
    .withColumn(
        "patient_id",
        F.regexp_extract(
            F.col("_patient_ref"),
            r"urn:uuid:([a-fA-F0-9-]+)",
            1
        )
    )
    .drop("_encounter_id", "_patient_ref")
)

print("Encounter patient references repaired")

fixed_encounter_df.select(
    "encounter_id",
    "patient_id",
    "start_time"
).show(10, truncate=False)

# COMMAND ----------

# Cell 8 — Validate Encounter → Patient relationship

fixed_encounter_df.select(
    F.count("*").alias("total_records"),
    F.count(
        F.when(
            F.col("patient_id").isNull() |
            (F.col("patient_id") == ""),
            True
        )
    ).alias("null_or_empty_patient_id"),
    F.countDistinct("patient_id").alias("distinct_patient_ids")
).show()

# COMMAND ----------

# Cell 9 — Persist corrected Encounter table to Silver

(
    fixed_encounter_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(
        "healthcare_lakehouse.silver.encounter"
    )
)

# Refresh the table reference
encounter_df = spark.table(
    "healthcare_lakehouse.silver.encounter"
)

print("Silver Encounter table updated successfully.")
print(f"Records: {encounter_df.count():,}")

# COMMAND ----------

# Cell 10 — Patient Encounter Summary

encounter_summary_df = (
    encounter_df
    .groupBy("patient_id")
    .agg(
        F.countDistinct("encounter_id").alias("total_encounters"),
        F.min("start_time").alias("first_encounter_time"),
        F.max("start_time").alias("last_encounter_time")
    )
)

print("Encounter summary created")
print(f"Patients with encounters: {encounter_summary_df.count():,}")

encounter_summary_df.show(10, truncate=False)

# COMMAND ----------

# Cell 11 — Add encounter metrics to Patient 360

patient_360_df = (
    patient_360_df
    .join(
        encounter_summary_df,
        on="patient_id",
        how="left"
    )
    .fillna({
        "total_encounters": 0
    })
)

print("Patient 360 enriched with encounter metrics")

patient_360_df.select(
    "patient_id",
    "gender",
    "age",
    "total_encounters",
    "first_encounter_time",
    "last_encounter_time"
).show(10, truncate=False)

# COMMAND ----------

# Cell 12 — Condition Summary

condition_summary_df = (
    condition_df
    .groupBy("patient_id")
    .agg(
        F.countDistinct("condition_id").alias("total_conditions"),
        F.countDistinct("condition_code").alias("distinct_condition_types"),
        F.min("onset_date").alias("first_condition_time"),
        F.max("onset_date").alias("latest_condition_time")
    )
)

print("Condition summary created")
print(f"Patients with conditions: {condition_summary_df.count():,}")

condition_summary_df.show(10, truncate=False)

# COMMAND ----------

# Cell 13 — Enrich Patient 360 with Condition metrics

patient_360_df = (
    patient_360_df
    .join(
        condition_summary_df,
        on="patient_id",
        how="left"
    )
    .fillna({
        "total_conditions": 0,
        "distinct_condition_types": 0
    })
)

print("Patient 360 enriched with condition metrics")

patient_360_df.select(
    "patient_id",
    "gender",
    "age",
    "total_encounters",
    "total_conditions",
    "distinct_condition_types",
    "first_condition_time",
    "latest_condition_time"
).show(10, truncate=False)

# COMMAND ----------

# Cell 14 — Validate Condition patient references

print("Condition records:", condition_df.count())

condition_df.select(
    "patient_id",
    "condition_id",
    "encounter_id"
).show(10, truncate=False)

condition_df.select(
    F.count("*").alias("total_records"),
    F.count(
        F.when(
            F.col("patient_id").isNull() |
            (F.trim(F.col("patient_id")) == ""),
            True
        )
    ).alias("null_or_empty_patient_id"),
    F.countDistinct("patient_id").alias("distinct_patient_ids")
).show()

# COMMAND ----------

# Cell 15 — Repair Condition patient references using Encounter

condition_fixed_df = (
    condition_df
    .drop("patient_id")
    .join(
        encounter_df.select(
            F.col("encounter_id").alias("_encounter_id"),
            F.col("patient_id").alias("_patient_id")
        ),
        condition_df["encounter_id"] == F.col("_encounter_id"),
        "left"
    )
    .withColumn(
        "patient_id",
        F.col("_patient_id")
    )
    .drop("_encounter_id", "_patient_id")
)

print("Condition patient references repaired")

condition_fixed_df.select(
    "condition_id",
    "patient_id",
    "encounter_id"
).show(10, truncate=False)

# COMMAND ----------

# Cell 16 — Validate Condition patient references

condition_fixed_df.select(
    F.count("*").alias("total_records"),
    F.count(
        F.when(
            F.col("patient_id").isNull() |
            (F.trim(F.col("patient_id")) == ""),
            True
        )
    ).alias("null_or_empty_patient_id"),
    F.countDistinct("patient_id").alias("distinct_patient_ids")
).show()

# COMMAND ----------

# Cell 17 — Persist repaired Condition table to Silver

(
    condition_fixed_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(
        "healthcare_lakehouse.silver.condition"
    )
)

condition_df = spark.table(
    "healthcare_lakehouse.silver.condition"
)

print("Silver Condition table updated successfully.")
print(f"Records: {condition_df.count():,}")

# COMMAND ----------

# Cell 18 — Rebuild Condition Summary from corrected Silver table

condition_summary_df = (
    condition_df
    .groupBy("patient_id")
    .agg(
        F.countDistinct("condition_id").alias("total_conditions"),
        F.countDistinct("condition_code").alias("distinct_condition_types"),
        F.min("onset_date").alias("first_condition_time"),
        F.max("onset_date").alias("latest_condition_time")
    )
)

print("Condition summary rebuilt")
print(f"Patients with conditions: {condition_summary_df.count():,}")

condition_summary_df.show(10, truncate=False)

# COMMAND ----------

# Cell 19 — Enrich Patient 360 with corrected Condition metrics

patient_360_df = (
    patient_360_df
    .drop(
        "total_conditions",
        "distinct_condition_types",
        "first_condition_time",
        "latest_condition_time"
    )
    .join(
        condition_summary_df,
        on="patient_id",
        how="left"
    )
    .fillna({
        "total_conditions": 0,
        "distinct_condition_types": 0
    })
)

print("Patient 360 updated with condition metrics")

patient_360_df.select(
    "patient_id",
    "gender",
    "age",
    "total_encounters",
    "total_conditions",
    "distinct_condition_types",
    "first_condition_time",
    "latest_condition_time"
).show(10, truncate=False)

# COMMAND ----------

# Cell 20 — Observation Summary

observation_summary_df = (
    observation_df
    .groupBy("patient_id")
    .agg(
        F.countDistinct("observation_id").alias("total_observations"),
        F.countDistinct("observation_code").alias("distinct_observation_types"),
        F.min("observation_time").alias("first_observation_time"),
        F.max("observation_time").alias("latest_observation_time")
    )
)

print("Observation summary created")
print(f"Patients with observations: {observation_summary_df.count():,}")

observation_summary_df.show(10, truncate=False)

# COMMAND ----------

# Cell 21 — Enrich Patient 360 with Observation metrics

patient_360_df = (
    patient_360_df
    .drop(
        "total_observations",
        "distinct_observation_types",
        "first_observation_time",
        "latest_observation_time"
    )
    .join(
        observation_summary_df,
        on="patient_id",
        how="left"
    )
    .fillna({
        "total_observations": 0,
        "distinct_observation_types": 0
    })
)

print("Patient 360 enriched with observation metrics")

patient_360_df.select(
    "patient_id",
    "gender",
    "age",
    "total_encounters",
    "total_conditions",
    "total_observations",
    "distinct_observation_types",
    "first_observation_time",
    "latest_observation_time"
).show(10, truncate=False)

# COMMAND ----------

# Cell 22 — Observation patient reference validation

observation_df.select(
    "patient_id",
    "observation_id",
    "observation_code",
    "observation_time"
).show(10, truncate=False)

print("Observation patient reference check")

observation_df.select(
    F.count("*").alias("total_records"),
    F.sum(
        F.when(
            F.col("patient_id").isNull() | (F.trim(F.col("patient_id")) == ""),
            1
        ).otherwise(0)
    ).alias("null_or_empty_patient_id"),
    F.countDistinct("patient_id").alias("distinct_patient_ids")
).show()

# COMMAND ----------

# Cell 23 — Repair Observation patient references using Encounter

observation_df = (
    observation_df
    .drop("patient_id")
    .join(
        encounter_df
        .select(
            "encounter_id",
            "patient_id"
        )
        .dropDuplicates(["encounter_id"]),
        on="encounter_id",
        how="left"
    )
)

print("Observation patient references repaired")

observation_df.select(
    "observation_id",
    "patient_id",
    "encounter_id",
    "observation_code",
    "observation_time"
).show(10, truncate=False)

# COMMAND ----------

# Cell 24 — Observation patient reference validation

observation_df.select(
    F.count("*").alias("total_records"),
    F.sum(
        F.when(
            F.col("patient_id").isNull() | (F.trim(F.col("patient_id")) == ""),
            1
        ).otherwise(0)
    ).alias("null_or_empty_patient_id"),
    F.countDistinct("patient_id").alias("distinct_patient_ids")
).show()


# COMMAND ----------

# Cell 25 — Update Silver Observation table

observation_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("healthcare_lakehouse.silver.observation")

print("Silver Observation table updated successfully.")
print(f"Records: {observation_df.count():,}")

# COMMAND ----------

# Cell 26 — Rebuild Observation Summary

observation_summary_df = (
    observation_df
    .groupBy("patient_id")
    .agg(
        F.countDistinct("observation_id").alias("total_observations"),
        F.countDistinct("observation_code").alias("distinct_observation_types"),
        F.min("observation_time").alias("first_observation_time"),
        F.max("observation_time").alias("latest_observation_time")
    )
)

print("Observation summary rebuilt")
print(f"Patients with observations: {observation_summary_df.count():,}")

observation_summary_df.show(10, truncate=False)

# COMMAND ----------

# Cell 27 — Enrich Patient 360 with corrected Observation metrics

patient_360_df = (
    patient_360_df
    .drop(
        "total_observations",
        "distinct_observation_types",
        "first_observation_time",
        "latest_observation_time"
    )
    .join(
        observation_summary_df,
        on="patient_id",
        how="left"
    )
    .fillna({
        "total_observations": 0,
        "distinct_observation_types": 0
    })
)

print("Patient 360 enriched with corrected observation metrics")

patient_360_df.select(
    "patient_id",
    "gender",
    "age",
    "total_encounters",
    "total_conditions",
    "total_observations",
    "distinct_observation_types",
    "first_observation_time",
    "latest_observation_time"
).show(10, truncate=False)

# COMMAND ----------

# Cell 28 — Procedure Summary

procedure_summary_df = (
    procedure_df
    .groupBy("patient_id")
    .agg(
        F.countDistinct("procedure_id").alias("total_procedures"),
        F.countDistinct("procedure_code").alias("distinct_procedure_types"),
        F.min("procedure_start").alias("first_procedure_time"),
        F.max("procedure_end").alias("latest_procedure_time")
    )
)

print("Procedure summary created")
print(f"Patients with procedures: {procedure_summary_df.count():,}")

procedure_summary_df.show(10, truncate=False)

# COMMAND ----------

# Cell 29 — Procedure patient reference validation

procedure_df.select(
    "patient_id",
    "procedure_id",
    "procedure_code",
    "encounter_id",
    "procedure_start",
    "procedure_end"
).show(10, truncate=False)

print("Procedure patient reference check")

procedure_df.select(
    F.count("*").alias("total_records"),
    F.sum(
        F.when(
            F.col("patient_id").isNull() | (F.trim(F.col("patient_id")) == ""),
            1
        ).otherwise(0)
    ).alias("null_or_empty_patient_id"),
    F.countDistinct("patient_id").alias("distinct_patient_ids")
).show()

# COMMAND ----------

# Cell 30 — Repair Procedure patient references using Encounter

procedure_df = (
    procedure_df
    .drop("patient_id")
    .join(
        encounter_df
        .select(
            "encounter_id",
            "patient_id"
        )
        .dropDuplicates(["encounter_id"]),
        on="encounter_id",
        how="left"
    )
)

print("Procedure patient references repaired")

procedure_df.select(
    "procedure_id",
    "patient_id",
    "encounter_id",
    "procedure_code",
    "procedure_start",
    "procedure_end"
).show(10, truncate=False)

# COMMAND ----------

# Cell 31 — Procedure patient reference validation

procedure_df.select(
    F.count("*").alias("total_records"),
    F.sum(
        F.when(
            F.col("patient_id").isNull() | (F.trim(F.col("patient_id")) == ""),
            1
        ).otherwise(0)
    ).alias("null_or_empty_patient_id"),
    F.countDistinct("patient_id").alias("distinct_patient_ids")
).show()

# COMMAND ----------

# Cell 32 — Update Silver Procedure table

procedure_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("healthcare_lakehouse.silver.procedure")

print("Silver Procedure table updated successfully.")
print(f"Records: {procedure_df.count():,}")

# COMMAND ----------

# Cell 33 — Rebuild Procedure Summary

procedure_summary_df = (
    procedure_df
    .groupBy("patient_id")
    .agg(
        F.countDistinct("procedure_id").alias("total_procedures"),
        F.countDistinct("procedure_code").alias("distinct_procedure_types"),
        F.min("procedure_start").alias("first_procedure_time"),
        F.max("procedure_end").alias("latest_procedure_time")
    )
)

print("Procedure summary rebuilt")
print(f"Patients with procedures: {procedure_summary_df.count():,}")

procedure_summary_df.show(10, truncate=False)

# COMMAND ----------

# CELL 34A — FIND MEDICATION REQUEST DATAFRAME

[name for name in dir() if "medication" in name.lower()]

# COMMAND ----------

# ============================================================
# CELL 34 — MEDICATION REQUEST SUMMARY
# ============================================================

from pyspark.sql import functions as F

medication_summary_df = (
    medication_request_df
    .filter(F.col("patient_id").isNotNull())
    .groupBy("patient_id")
    .agg(
        F.count("medication_request_id").alias("total_medication_requests"),
        F.countDistinct("medication_name").alias("distinct_medications"),
        F.min("authored_on").alias("first_medication_time"),
        F.max("authored_on").alias("latest_medication_time")
    )
)

print(
    "Medication summary created | "
    f"Patients with medications: "
    f"{medication_summary_df.select('patient_id').distinct().count():,}"
)

display(
    medication_summary_df.orderBy("patient_id")
)

# COMMAND ----------

# ============================================================
# CELL 35 — REPAIR MEDICATION REQUEST PATIENT REFERENCES
# ============================================================

medication_request_df = (
    medication_request_df
    .drop("patient_id")
    .join(
        encounter_df
        .select(
            "encounter_id",
            "patient_id"
        )
        .dropDuplicates(["encounter_id"]),
        on="encounter_id",
        how="left"
    )
)

print("MedicationRequest patient references repaired")

medication_request_df.select(
    "medication_request_id",
    "patient_id",
    "encounter_id"
).show(10, truncate=False)

medication_request_df.agg(
    F.count("*").alias("total_records"),
    F.sum(
        F.when(
            F.col("patient_id").isNull() |
            (F.trim(F.col("patient_id")) == ""),
            1
        ).otherwise(0)
    ).alias("null_or_empty_patient_id"),
    F.countDistinct("patient_id").alias("distinct_patient_ids")
).show()

# COMMAND ----------

# ============================================================
# CELL 36 — REBUILD MEDICATION REQUEST SUMMARY
# ============================================================

medication_summary_df = (
    medication_request_df
    .filter(F.col("patient_id").isNotNull())
    .groupBy("patient_id")
    .agg(
        F.count("medication_request_id").alias("total_medication_requests"),
        F.countDistinct("medication_name").alias("distinct_medications"),
        F.min("authored_on").alias("first_medication_time"),
        F.max("authored_on").alias("latest_medication_time")
    )
)

print(
    "Medication summary rebuilt | "
    f"Patients with medications: "
    f"{medication_summary_df.select('patient_id').distinct().count():,}"
)

display(
    medication_summary_df.orderBy("patient_id")
)

# COMMAND ----------

# ============================================================
# CELL 37 — REBUILD PATIENT 360 CLEANLY
# ============================================================

from pyspark.sql import functions as F

# ------------------------------------------------------------
# 1. Patient foundation
# ------------------------------------------------------------

patient_360_df = (
    patient_df
    .select(
        "patient_id",
        "birth_date",
        "gender",
        "marital_status",
        "multiple_birth"
    )
    .withColumn(
        "birth_year",
        F.year(F.to_date("birth_date"))
    )
    .withColumn(
        "age",
        F.floor(
            F.months_between(
                F.current_date(),
                F.to_date("birth_date")
            ) / 12
        ).cast("int")
    )
)

# ------------------------------------------------------------
# 2. Encounter metrics
# ------------------------------------------------------------

patient_360_df = patient_360_df.join(
    encounter_summary_df,
    on="patient_id",
    how="left"
)

# ------------------------------------------------------------
# 3. Condition metrics
# ------------------------------------------------------------

patient_360_df = patient_360_df.join(
    condition_summary_df,
    on="patient_id",
    how="left"
)

# ------------------------------------------------------------
# 4. Observation metrics
# ------------------------------------------------------------

patient_360_df = patient_360_df.join(
    observation_summary_df,
    on="patient_id",
    how="left"
)

# ------------------------------------------------------------
# 5. Medication metrics
# ------------------------------------------------------------

patient_360_df = patient_360_df.join(
    medication_summary_df,
    on="patient_id",
    how="left"
)

# ------------------------------------------------------------
# 6. Fill missing count metrics
# ------------------------------------------------------------

patient_360_df = patient_360_df.fillna({
    "total_encounters": 0,
    "total_conditions": 0,
    "distinct_condition_types": 0,
    "total_observations": 0,
    "distinct_observation_types": 0,
    "total_medication_requests": 0,
    "distinct_medications": 0
})

print("Patient 360 rebuilt successfully")

display(
    patient_360_df.select(
        "patient_id",
        "birth_date",
        "birth_year",
        "gender",
        "age",
        "total_encounters",
        "total_conditions",
        "total_observations",
        "total_medication_requests",
        "distinct_medications",
        "first_medication_time",
        "latest_medication_time"
    )
    .orderBy("patient_id")
)

# COMMAND ----------

# ============================================================
# CELL 38 — PATIENT 360 VALIDATION
# ============================================================

print("=" * 70)
print("PATIENT 360 GOLD VALIDATION")
print("=" * 70)

print(
    "Total patients:",
    patient_360_df.select("patient_id").distinct().count()
)

print(
    "Total rows:",
    patient_360_df.count()
)

print(
    "Duplicate patient IDs:",
    patient_360_df.groupBy("patient_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

print("\nSchema:")
patient_360_df.printSchema()

display(
    patient_360_df.select(
        "patient_id",
        "gender",
        "age",
        "total_encounters",
        "total_conditions",
        "total_observations",
        "total_medication_requests"
    )
    .orderBy("patient_id")
)

# COMMAND ----------

# ============================================================
# CELL 39 — PROCEDURE METRICS ENRICHMENT
# ============================================================

patient_360_df = (
    patient_360_df
    .join(
        procedure_summary_df,
        on="patient_id",
        how="left"
    )
    .fillna({
        "total_procedures": 0,
        "distinct_procedure_types": 0
    })
)

print("Patient 360 enriched with procedure metrics")

display(
    patient_360_df.select(
        "patient_id",
        "gender",
        "age",
        "total_encounters",
        "total_conditions",
        "total_observations",
        "total_procedures",
        "distinct_procedure_types",
        "first_procedure_time",
        "latest_procedure_time"
    )
    .orderBy("patient_id")
)

# COMMAND ----------

# ============================================================
# CELL 40 — DIAGNOSTIC REPORT SUMMARY
# ============================================================

diagnostic_summary_df = (
    diagnostic_report_df
    .filter(F.col("patient_id").isNotNull())
    .groupBy("patient_id")
    .agg(
        F.count("diagnostic_report_id").alias(
            "total_diagnostic_reports"
        ),

        F.countDistinct("diagnostic_code").alias(
            "distinct_diagnostic_types"
        ),

        F.min(
            F.coalesce(
                F.col("effective_time"),
                F.col("issued_time")
            )
        ).alias("first_diagnostic_time"),

        F.max(
            F.coalesce(
                F.col("effective_time"),
                F.col("issued_time")
            )
        ).alias("latest_diagnostic_time")
    )
)

print(
    "Diagnostic Report summary created | "
    f"Patients with reports: "
    f"{diagnostic_summary_df.select('patient_id').distinct().count():,}"
)

display(
    diagnostic_summary_df.orderBy("patient_id")
)

# COMMAND ----------

# ============================================================
# CELL 41 — DIAGNOSTIC METRICS ENRICHMENT
# ============================================================

patient_360_df = (
    patient_360_df
    .join(
        diagnostic_summary_df,
        on="patient_id",
        how="left"
    )
    .fillna({
        "total_diagnostic_reports": 0,
        "distinct_diagnostic_types": 0
    })
)

print("Patient 360 enriched with diagnostic metrics")

display(
    patient_360_df.select(
        "patient_id",
        "gender",
        "age",
        "total_encounters",
        "total_conditions",
        "total_observations",
        "total_procedures",
        "total_medication_requests",
        "total_diagnostic_reports",
        "distinct_diagnostic_types",
        "first_diagnostic_time",
        "latest_diagnostic_time"
    )
    .orderBy("patient_id")
)

# COMMAND ----------

# ============================================================
# CELL 42 — IMMUNIZATION SUMMARY
# ============================================================

immunization_summary_df = (
    immunization_df
    .filter(F.col("patient_id").isNotNull())
    .groupBy("patient_id")
    .agg(
        F.count("immunization_id").alias(
            "total_immunizations"
        ),

        F.countDistinct("vaccine_code").alias(
            "distinct_vaccine_types"
        ),

        F.min("vaccination_time").alias(
            "first_immunization_time"
        ),

        F.max("vaccination_time").alias(
            "latest_immunization_time"
        )
    )
)

print(
    "Immunization summary created | "
    f"Patients with immunizations: "
    f"{immunization_summary_df.select('patient_id').distinct().count():,}"
)

display(
    immunization_summary_df.orderBy("patient_id")
)

# COMMAND ----------

# ============================================================
# CELL 43 — IMMUNIZATION METRICS ENRICHMENT
# ============================================================

patient_360_df = (
    patient_360_df
    .join(
        immunization_summary_df,
        on="patient_id",
        how="left"
    )
    .fillna({
        "total_immunizations": 0,
        "distinct_vaccine_types": 0
    })
)

print("Patient 360 enriched with immunization metrics")

display(
    patient_360_df.select(
        "patient_id",
        "gender",
        "age",
        "total_encounters",
        "total_conditions",
        "total_observations",
        "total_procedures",
        "total_medication_requests",
        "total_diagnostic_reports",
        "total_immunizations",
        "distinct_vaccine_types",
        "first_immunization_time",
        "latest_immunization_time"
    )
    .orderBy("patient_id")
)

# COMMAND ----------

# ============================================================
# CELL 44 — CLINICAL ACTIVITY METRICS
# ============================================================

patient_360_df = (
    patient_360_df
    .withColumn(
        "total_clinical_events",
        F.coalesce(F.col("total_encounters"), F.lit(0))
        + F.coalesce(F.col("total_conditions"), F.lit(0))
        + F.coalesce(F.col("total_observations"), F.lit(0))
        + F.coalesce(F.col("total_procedures"), F.lit(0))
        + F.coalesce(F.col("total_medication_requests"), F.lit(0))
        + F.coalesce(F.col("total_diagnostic_reports"), F.lit(0))
        + F.coalesce(F.col("total_immunizations"), F.lit(0))
    )
)

print("Clinical activity metrics created")

display(
    patient_360_df.select(
        "patient_id",
        "age",
        "total_encounters",
        "total_conditions",
        "total_observations",
        "total_procedures",
        "total_medication_requests",
        "total_diagnostic_reports",
        "total_immunizations",
        "total_clinical_events"
    )
    .orderBy(F.desc("total_clinical_events"))
)

# COMMAND ----------

# ============================================================
# CELL 45 — PATIENT ACTIVITY CLASSIFICATION
# ============================================================

patient_360_df = (
    patient_360_df
    .withColumn(
        "activity_level",
        F.when(
            F.col("total_clinical_events") >= 500,
            "HIGH"
        )
        .when(
            F.col("total_clinical_events") >= 100,
            "MEDIUM"
        )
        .otherwise("LOW")
    )
)

print("Patient activity classification created")

display(
    patient_360_df.select(
        "patient_id",
        "age",
        "total_clinical_events",
        "activity_level"
    )
    .orderBy(
        F.desc("total_clinical_events")
    )
)

# COMMAND ----------

# ============================================================
# CELL 46 — PATIENT 360 UTILIZATION METRICS
# ============================================================

patient_360_df = (
    patient_360_df
    .withColumn(
        "conditions_per_encounter",
        F.when(
            F.col("total_encounters") > 0,
            F.round(
                F.col("total_conditions") /
                F.col("total_encounters"),
                2
            )
        ).otherwise(F.lit(0.0))
    )
    .withColumn(
        "observations_per_encounter",
        F.when(
            F.col("total_encounters") > 0,
            F.round(
                F.col("total_observations") /
                F.col("total_encounters"),
                2
            )
        ).otherwise(F.lit(0.0))
    )
    .withColumn(
        "procedures_per_encounter",
        F.when(
            F.col("total_encounters") > 0,
            F.round(
                F.col("total_procedures") /
                F.col("total_encounters"),
                2
            )
        ).otherwise(F.lit(0.0))
    )
    .withColumn(
        "medications_per_encounter",
        F.when(
            F.col("total_encounters") > 0,
            F.round(
                F.col("total_medication_requests") /
                F.col("total_encounters"),
                2
            )
        ).otherwise(F.lit(0.0))
    )
)

print("Patient 360 utilization metrics created")

display(
    patient_360_df.select(
        "patient_id",
        "total_encounters",
        "total_conditions",
        "total_observations",
        "total_procedures",
        "total_medication_requests",
        "conditions_per_encounter",
        "observations_per_encounter",
        "procedures_per_encounter",
        "medications_per_encounter"
    )
    .orderBy(F.desc("total_clinical_events"))
)

# COMMAND ----------

# ============================================================
# CELL 47 — GOLD PATIENT 360 QUALITY VALIDATION
# ============================================================

print("=" * 70)
print("GOLD PATIENT 360 — QUALITY VALIDATION")
print("=" * 70)

total_rows = patient_360_df.count()

distinct_patients = (
    patient_360_df
    .select("patient_id")
    .distinct()
    .count()
)

duplicate_patients = (
    patient_360_df
    .groupBy("patient_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

null_patient_ids = (
    patient_360_df
    .filter(
        F.col("patient_id").isNull() |
        (F.trim(F.col("patient_id")) == "")
    )
    .count()
)

print(f"Total rows              : {total_rows:,}")
print(f"Distinct patient IDs    : {distinct_patients:,}")
print(f"Duplicate patient IDs   : {duplicate_patients:,}")
print(f"Null/empty patient IDs  : {null_patient_ids:,}")

print("\nExpected:")
print("Total rows              : 109")
print("Distinct patient IDs    : 109")
print("Duplicate patient IDs   : 0")
print("Null/empty patient IDs  : 0")

print("\nGold Patient 360 schema:")
patient_360_df.printSchema()

# COMMAND ----------

# ============================================================
# CELL 48 — FINAL GOLD PATIENT 360
# ============================================================

gold_patient_360_df = patient_360_df.select(
    "patient_id",
    "birth_date",
    "birth_year",
    "gender",
    "marital_status",
    "multiple_birth",
    "age",

    # Encounter
    "total_encounters",
    "first_encounter_time",
    "last_encounter_time",

    # Conditions
    "total_conditions",
    "distinct_condition_types",
    "first_condition_time",
    "latest_condition_time",

    # Observations
    "total_observations",
    "distinct_observation_types",
    "first_observation_time",
    "latest_observation_time",

    # Procedures
    "total_procedures",
    "distinct_procedure_types",
    "first_procedure_time",
    "latest_procedure_time",

    # Medications
    "total_medication_requests",
    "distinct_medications",
    "first_medication_time",
    "latest_medication_time",

    # Diagnostics
    "total_diagnostic_reports",
    "distinct_diagnostic_types",
    "first_diagnostic_time",
    "latest_diagnostic_time",

    # Immunizations
    "total_immunizations",
    "distinct_vaccine_types",
    "first_immunization_time",
    "latest_immunization_time",

    # Overall utilization
    "total_clinical_events",
    "activity_level",
    "conditions_per_encounter",
    "observations_per_encounter",
    "procedures_per_encounter",
    "medications_per_encounter"
)

print("Final Gold Patient 360 projection created")

display(
    gold_patient_360_df.orderBy("patient_id")
)

# COMMAND ----------

# ============================================================
# CELL 49 — FINAL GOLD VALIDATION
# ============================================================

print("=" * 70)
print("FINAL GOLD PATIENT 360 VALIDATION")
print("=" * 70)

total_rows = gold_patient_360_df.count()

distinct_patients = (
    gold_patient_360_df
    .select("patient_id")
    .distinct()
    .count()
)

duplicate_patients = (
    gold_patient_360_df
    .groupBy("patient_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

null_patient_ids = (
    gold_patient_360_df
    .filter(F.col("patient_id").isNull())
    .count()
)

print(f"Total rows             : {total_rows:,}")
print(f"Distinct patients      : {distinct_patients:,}")
print(f"Duplicate patient IDs  : {duplicate_patients:,}")
print(f"Null patient IDs       : {null_patient_ids:,}")

assert total_rows == 109
assert distinct_patients == 109
assert duplicate_patients == 0
assert null_patient_ids == 0

print("\n✓ GOLD PATIENT 360 VALIDATED")

# COMMAND ----------

# ============================================================
# CELL 50 — WRITE FINAL GOLD TABLE
# ============================================================

gold_table_name = "healthcare_lakehouse.gold.patient_360"

(
    gold_patient_360_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(gold_table_name)
)

print("=" * 70)
print("GOLD TABLE CREATED SUCCESSFULLY")
print("=" * 70)
print(f"Table: {gold_table_name}")

display(
    spark.table(gold_table_name)
    .limit(10)
)

# COMMAND ----------

# ============================================================
# CELL 51 — FINAL GOLD TABLE CHECK
# ============================================================

final_gold_df = spark.table(
    "healthcare_lakehouse.gold.patient_360"
)

print("=" * 70)
print("GOLD LAYER COMPLETE")
print("=" * 70)

print(f"Records : {final_gold_df.count():,}")
print(
    f"Patients: "
    f"{final_gold_df.select('patient_id').distinct().count():,}"
)

print("\nFinal Gold table:")
display(
    final_gold_df.orderBy("patient_id").limit(10)
)

# COMMAND ----------

