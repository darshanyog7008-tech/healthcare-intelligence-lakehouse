# Databricks notebook source
# ============================================================
# CELL 1 — DATA QUALITY SETUP
# ============================================================

from pyspark.sql import functions as F

gold_df = spark.table(
    "healthcare_lakehouse.gold.patient_360"
)

print("Gold Patient 360 loaded")
print(f"Records: {gold_df.count():,}")

display(gold_df.limit(10))

# COMMAND ----------

# ============================================================
# CELL 2 — ROW COUNT & PRIMARY KEY VALIDATION
# ============================================================

total_rows = gold_df.count()

distinct_patient_ids = (
    gold_df
    .select("patient_id")
    .distinct()
    .count()
)

duplicate_patient_ids = (
    gold_df
    .groupBy("patient_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

print("=" * 60)
print("PATIENT 360 — PRIMARY KEY VALIDATION")
print("=" * 60)

print(f"Total rows             : {total_rows:,}")
print(f"Distinct patient IDs   : {distinct_patient_ids:,}")
print(f"Duplicate patient IDs  : {duplicate_patient_ids:,}")

if total_rows == distinct_patient_ids and duplicate_patient_ids == 0:
    print("\n✓ PRIMARY KEY VALIDATION PASSED")
else:
    print("\n✗ PRIMARY KEY VALIDATION FAILED")

# COMMAND ----------

# ============================================================
# CELL 3 — NULL & EMPTY VALUE VALIDATION
# ============================================================

columns_to_check = [
    "patient_id",
    "gender",
    "birth_date",
    "age",
    "total_encounters",
    "total_conditions",
    "total_observations",
    "total_procedures",
    "total_medication_requests"
]

null_results = []

for column_name in columns_to_check:
    null_count = (
        gold_df
        .filter(
            F.col(column_name).isNull() |
            (
                F.col(column_name).cast("string").isNotNull() &
                (F.trim(F.col(column_name).cast("string")) == "")
            )
        )
        .count()
    )

    null_results.append(
        (column_name, null_count)
    )

null_df = spark.createDataFrame(
    null_results,
    ["column_name", "null_or_empty_count"]
)

display(
    null_df.orderBy(F.desc("null_or_empty_count"))
)

# COMMAND ----------

# ============================================================
# CELL 4 — INVALID NEGATIVE METRIC VALIDATION
# ============================================================

metric_columns = [
    "total_encounters",
    "total_conditions",
    "total_observations",
    "total_procedures",
    "total_medication_requests",
    "total_diagnostic_reports",
    "total_immunizations",
    "total_clinical_events"
]

invalid_metric_results = []

for column_name in metric_columns:
    invalid_count = (
        gold_df
        .filter(F.col(column_name) < 0)
        .count()
    )

    invalid_metric_results.append(
        (column_name, invalid_count)
    )

invalid_metrics_df = spark.createDataFrame(
    invalid_metric_results,
    ["metric", "negative_value_count"]
)

display(
    invalid_metrics_df
)

# COMMAND ----------

# ============================================================
# CELL 5 — AGE VALIDATION
# ============================================================

invalid_age_df = (
    gold_df
    .filter(
        F.col("age").isNotNull() &
        (
            (F.col("age") < 0) |
            (F.col("age") > 120)
        )
    )
    .select(
        "patient_id",
        "birth_date",
        "age"
    )
)

invalid_age_count = invalid_age_df.count()

print("=" * 60)
print("AGE DATA QUALITY CHECK")
print("=" * 60)

print(f"Invalid age records: {invalid_age_count:,}")

if invalid_age_count == 0:
    print("✓ AGE VALIDATION PASSED")
else:
    print("✗ AGE VALIDATION FAILED")
    display(invalid_age_df)

# COMMAND ----------

# ============================================================
# CELL 6 — CLINICAL TIMESTAMP VALIDATION
# ============================================================

timestamp_pairs = [
    ("first_encounter_time", "last_encounter_time"),
    ("first_condition_time", "latest_condition_time"),
    ("first_observation_time", "latest_observation_time"),
    ("first_procedure_time", "latest_procedure_time"),
    ("first_medication_time", "latest_medication_time"),
    ("first_diagnostic_time", "latest_diagnostic_time"),
    ("first_immunization_time", "latest_immunization_time")
]

invalid_timestamp_results = []

for first_col, latest_col in timestamp_pairs:

    invalid_count = (
        gold_df
        .filter(
            F.col(first_col).isNotNull() &
            F.col(latest_col).isNotNull() &
            (F.to_timestamp(F.col(first_col)) >
             F.to_timestamp(F.col(latest_col)))
        )
        .count()
    )

    invalid_timestamp_results.append(
        (first_col, latest_col, invalid_count)
    )

timestamp_quality_df = spark.createDataFrame(
    invalid_timestamp_results,
    ["first_timestamp", "latest_timestamp", "invalid_order_count"]
)

display(timestamp_quality_df)

# COMMAND ----------

# ============================================================
# CELL 7 — CLINICAL METRIC CONSISTENCY
# ============================================================

metric_check_df = (
    gold_df
    .withColumn(
        "calculated_clinical_events",
        F.coalesce(F.col("total_encounters"), F.lit(0)) +
        F.coalesce(F.col("total_conditions"), F.lit(0)) +
        F.coalesce(F.col("total_observations"), F.lit(0)) +
        F.coalesce(F.col("total_procedures"), F.lit(0)) +
        F.coalesce(F.col("total_medication_requests"), F.lit(0)) +
        F.coalesce(F.col("total_diagnostic_reports"), F.lit(0)) +
        F.coalesce(F.col("total_immunizations"), F.lit(0))
    )
    .withColumn(
        "difference",
        F.col("total_clinical_events") -
        F.col("calculated_clinical_events")
    )
)

inconsistent_count = (
    metric_check_df
    .filter(F.col("difference") != 0)
    .count()
)

print("=" * 60)
print("CLINICAL EVENT CONSISTENCY CHECK")
print("=" * 60)

print(f"Inconsistent records: {inconsistent_count:,}")

if inconsistent_count == 0:
    print("✓ CLINICAL EVENT CONSISTENCY PASSED")
else:
    print("✗ CLINICAL EVENT CONSISTENCY FAILED")

    display(
        metric_check_df
        .filter(F.col("difference") != 0)
        .select(
            "patient_id",
            "total_clinical_events",
            "calculated_clinical_events",
            "difference"
        )
    )

# COMMAND ----------

# ============================================================
# CELL 8 — UTILIZATION METRIC CONSISTENCY
# ============================================================

utilization_check_df = (
    gold_df
    .withColumn(
        "expected_conditions_per_encounter",
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
        "expected_observations_per_encounter",
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
        "expected_procedures_per_encounter",
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
        "expected_medications_per_encounter",
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

inconsistent_utilization_count = (
    utilization_check_df
    .filter(
        (F.col("conditions_per_encounter") !=
         F.col("expected_conditions_per_encounter")) |
        (F.col("observations_per_encounter") !=
         F.col("expected_observations_per_encounter")) |
        (F.col("procedures_per_encounter") !=
         F.col("expected_procedures_per_encounter")) |
        (F.col("medications_per_encounter") !=
         F.col("expected_medications_per_encounter"))
    )
    .count()
)

print("=" * 60)
print("UTILIZATION METRIC CONSISTENCY CHECK")
print("=" * 60)

print(
    f"Inconsistent records: "
    f"{inconsistent_utilization_count:,}"
)

if inconsistent_utilization_count == 0:
    print("✓ UTILIZATION METRICS PASSED")
else:
    print("✗ UTILIZATION METRICS FAILED")

# COMMAND ----------

# ============================================================
# CELL 9 — DEMOGRAPHIC CONSISTENCY VALIDATION
# ============================================================

valid_genders = ["male", "female", "other", "unknown"]

invalid_gender_df = (
    gold_df
    .filter(
        F.col("gender").isNotNull() &
        (~F.lower(F.trim(F.col("gender"))).isin(valid_genders))
    )
    .select(
        "patient_id",
        "gender"
    )
)

invalid_gender_count = invalid_gender_df.count()

invalid_multiple_birth_df = (
    gold_df
    .filter(
        F.col("multiple_birth").isNotNull() &
        (
            ~F.col("multiple_birth").cast("string").isin(
                ["true", "false", "0", "1"]
            )
        )
    )
    .select(
        "patient_id",
        "multiple_birth"
    )
)

invalid_multiple_birth_count = invalid_multiple_birth_df.count()

print("=" * 60)
print("DEMOGRAPHIC CONSISTENCY CHECK")
print("=" * 60)

print(f"Invalid gender records         : {invalid_gender_count:,}")
print(f"Invalid multiple-birth records: {invalid_multiple_birth_count:,}")

if invalid_gender_count == 0 and invalid_multiple_birth_count == 0:
    print("\n✓ DEMOGRAPHIC VALIDATION PASSED")
else:
    print("\n✗ DEMOGRAPHIC VALIDATION FAILED")

# COMMAND ----------

# ============================================================
# CELL 10 — DUPLICATE RECORD DETECTION
# ============================================================

duplicate_df = (
    gold_df
    .groupBy("patient_id")
    .count()
    .filter(F.col("count") > 1)
)

duplicate_count = duplicate_df.count()

print("=" * 60)
print("DUPLICATE PATIENT RECORD CHECK")
print("=" * 60)

print(f"Duplicate patient IDs: {duplicate_count:,}")

if duplicate_count == 0:
    print("✓ DUPLICATE CHECK PASSED")
else:
    print("✗ DUPLICATE CHECK FAILED")

    display(
        duplicate_df.orderBy(F.desc("count"))
    )

# COMMAND ----------

# ============================================================
# CELL 11 — CONSOLIDATED DATA QUALITY REPORT
# ============================================================

total_rows = gold_df.count()

distinct_patients = (
    gold_df
    .select("patient_id")
    .distinct()
    .count()
)

null_patient_ids = (
    gold_df
    .filter(
        F.col("patient_id").isNull() |
        (F.trim(F.col("patient_id")) == "")
    )
    .count()
)

duplicate_patients = (
    gold_df
    .groupBy("patient_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

negative_metric_rows = (
    gold_df
    .filter(
        (F.col("total_encounters") < 0) |
        (F.col("total_conditions") < 0) |
        (F.col("total_observations") < 0) |
        (F.col("total_procedures") < 0) |
        (F.col("total_medication_requests") < 0)
    )
    .count()
)

invalid_age_rows = (
    gold_df
    .filter(
        F.col("age").isNotNull() &
        (
            (F.col("age") < 0) |
            (F.col("age") > 120)
        )
    )
    .count()
)

quality_report = spark.createDataFrame(
    [
        ("row_count", total_rows, "PASS" if total_rows > 0 else "FAIL"),
        ("distinct_patient_ids", distinct_patients,
         "PASS" if distinct_patients == total_rows else "FAIL"),
        ("null_or_empty_patient_ids", null_patient_ids,
         "PASS" if null_patient_ids == 0 else "FAIL"),
        ("duplicate_patient_ids", duplicate_patients,
         "PASS" if duplicate_patients == 0 else "FAIL"),
        ("negative_metric_rows", negative_metric_rows,
         "PASS" if negative_metric_rows == 0 else "FAIL"),
        ("invalid_age_rows", invalid_age_rows,
         "PASS" if invalid_age_rows == 0 else "FAIL")
    ],
    ["check_name", "record_count", "status"]
)

display(quality_report)

# COMMAND ----------

# ============================================================
# CELL 12 — FINAL DATA QUALITY GATE
# ============================================================

failed_checks = (
    quality_report
    .filter(F.col("status") == "FAIL")
    .count()
)

print("=" * 60)
print("FINAL DATA QUALITY GATE")
print("=" * 60)

print(f"Total checks : {quality_report.count()}")
print(f"Failed checks: {failed_checks}")

if failed_checks == 0:
    print("\n✓ DATA QUALITY PASSED")
    print("✓ Gold Patient 360 is ready for downstream analytics.")
else:
    print("\n✗ DATA QUALITY FAILED")
    print("✗ Gold Patient 360 requires investigation.")
    
    display(
        quality_report
        .filter(F.col("status") == "FAIL")
    )

# COMMAND ----------

