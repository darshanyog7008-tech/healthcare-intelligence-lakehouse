# Databricks notebook source
# ============================================================
# CELL 1 — HEALTHCARE INSIGHTS DATASET
# ============================================================

from pyspark.sql import functions as F

# Load Gold Patient 360
patient_360_df = spark.table(
    "healthcare_lakehouse.gold.patient_360"
)

# Load RAG corpus created in the previous notebook
rag_corpus_df = spark.table(
    "healthcare_lakehouse.gold.rag_corpus"
)

print("=" * 60)
print("AI HEALTHCARE INSIGHTS")
print("=" * 60)

print(f"Patient 360 records : {patient_360_df.count():,}")
print(f"RAG documents       : {rag_corpus_df.count():,}")

display(
    patient_360_df.limit(5)
)

# COMMAND ----------

# ============================================================
# CELL 2 — PATIENT CLINICAL ACTIVITY PROFILE
# ============================================================

clinical_activity_df = (
    patient_360_df
    .select(
        "patient_id",
        "age",
        "gender",
        "total_encounters",
        "total_conditions",
        "total_observations",
        "total_procedures",
        "total_medication_requests",
        "total_clinical_events"
    )
    .withColumn(
        "clinical_events_per_encounter",
        F.when(
            F.col("total_encounters") > 0,
            F.round(
                F.col("total_clinical_events") /
                F.col("total_encounters"),
                2
            )
        ).otherwise(F.lit(0.0))
    )
)

print("=" * 60)
print("PATIENT CLINICAL ACTIVITY PROFILE")
print("=" * 60)

display(
    clinical_activity_df.orderBy(
        F.desc("total_clinical_events")
    ).limit(10)
)

# COMMAND ----------

# ============================================================
# CELL 3 — CONDITION INSIGHTS
# ============================================================

condition_insights_df = (
    patient_360_df
    .select(
        "patient_id",
        "age",
        "gender",
        "total_conditions",
        "distinct_condition_types"
    )
    .withColumn(
        "condition_diversity_ratio",
        F.when(
            F.col("total_conditions") > 0,
            F.round(
                F.col("distinct_condition_types") /
                F.col("total_conditions"),
                3
            )
        ).otherwise(F.lit(0.0))
    )
    .withColumn(
        "condition_activity_level",
        F.when(F.col("total_conditions") == 0, "No recorded conditions")
         .when(F.col("total_conditions") <= 3, "Low")
         .when(F.col("total_conditions") <= 10, "Moderate")
         .otherwise("High")
    )
)

print("=" * 60)
print("CONDITION INSIGHTS")
print("=" * 60)

display(
    condition_insights_df
    .orderBy(F.desc("total_conditions"))
    .limit(10)
)

# COMMAND ----------

# ============================================================
# CELL 4 — OBSERVATION INSIGHTS
# ============================================================

observation_insights_df = (
    patient_360_df
    .select(
        "patient_id",
        "age",
        "gender",
        "total_observations",
        "distinct_observation_types"
    )
    .withColumn(
        "observation_diversity_ratio",
        F.when(
            F.col("total_observations") > 0,
            F.round(
                F.col("distinct_observation_types") /
                F.col("total_observations"),
                3
            )
        ).otherwise(F.lit(0.0))
    )
    .withColumn(
        "observation_activity_level",
        F.when(F.col("total_observations") == 0, "No recorded observations")
         .when(F.col("total_observations") <= 10, "Low")
         .when(F.col("total_observations") <= 50, "Moderate")
         .otherwise("High")
    )
)

print("=" * 60)
print("OBSERVATION INSIGHTS")
print("=" * 60)

display(
    observation_insights_df
    .orderBy(F.desc("total_observations"))
    .limit(10)
)

# COMMAND ----------

# ============================================================
# CELL 5 — PROCEDURE INSIGHTS
# ============================================================

procedure_insights_df = (
    patient_360_df
    .select(
        "patient_id",
        "age",
        "gender",
        "total_procedures",
        "distinct_procedure_types"
    )
    .withColumn(
        "procedure_diversity_ratio",
        F.when(
            F.col("total_procedures") > 0,
            F.round(
                F.col("distinct_procedure_types") /
                F.col("total_procedures"),
                3
            )
        ).otherwise(F.lit(0.0))
    )
    .withColumn(
        "procedure_activity_level",
        F.when(F.col("total_procedures") == 0, "No recorded procedures")
         .when(F.col("total_procedures") <= 5, "Low")
         .when(F.col("total_procedures") <= 20, "Moderate")
         .otherwise("High")
    )
)

print("=" * 60)
print("PROCEDURE INSIGHTS")
print("=" * 60)

display(
    procedure_insights_df
    .orderBy(F.desc("total_procedures"))
    .limit(10)
)

# COMMAND ----------

# ============================================================
# CELL 6 — MEDICATION INSIGHTS
# ============================================================

medication_insights_df = (
    patient_360_df
    .select(
        "patient_id",
        "age",
        "gender",
        "total_medication_requests",
        "distinct_medications"
    )
    .withColumn(
        "medication_diversity_ratio",
        F.when(
            F.col("total_medication_requests") > 0,
            F.round(
                F.col("distinct_medications") /
                F.col("total_medication_requests"),
                3
            )
        ).otherwise(F.lit(0.0))
    )
    .withColumn(
        "medication_activity_level",
        F.when(
            F.col("total_medication_requests") == 0,
            "No recorded medication requests"
        )
        .when(
            F.col("total_medication_requests") <= 5,
            "Low"
        )
        .when(
            F.col("total_medication_requests") <= 20,
            "Moderate"
        )
        .otherwise("High")
    )
)

print("=" * 60)
print("MEDICATION INSIGHTS")
print("=" * 60)

display(
    medication_insights_df
    .orderBy(F.desc("total_medication_requests"))
    .limit(10)
)

# COMMAND ----------

# ============================================================
# CELL 7 — PATIENT 360 COMBINED INSIGHT PROFILE
# ============================================================

patient_insight_df = (
    clinical_activity_df
    .join(
        condition_insights_df.select(
            "patient_id",
            "distinct_condition_types",
            "condition_activity_level"
        ),
        on="patient_id",
        how="left"
    )
    .join(
        observation_insights_df.select(
            "patient_id",
            "distinct_observation_types",
            "observation_activity_level"
        ),
        on="patient_id",
        how="left"
    )
    .join(
        procedure_insights_df.select(
            "patient_id",
            "distinct_procedure_types",
            "procedure_activity_level"
        ),
        on="patient_id",
        how="left"
    )
    .join(
        medication_insights_df.select(
            "patient_id",
            "distinct_medications",
            "medication_activity_level"
        ),
        on="patient_id",
        how="left"
    )
)

print("=" * 60)
print("PATIENT 360 AI INSIGHT PROFILE")
print("=" * 60)

print(
    f"Patients available for insight generation: "
    f"{patient_insight_df.count():,}"
)

display(
    patient_insight_df
    .orderBy(F.desc("total_clinical_events"))
    .limit(10)
)

# COMMAND ----------

# ============================================================
# CELL 8 — GROUNDED HEALTHCARE INSIGHT PROMPT
# ============================================================

# Select one patient for the demonstration
selected_patient = (
    patient_insight_df
    .orderBy(F.desc("total_clinical_events"))
    .first()
)

patient_profile_text = f"""
Patient ID: {selected_patient['patient_id']}
Age: {selected_patient['age']}
Gender: {selected_patient['gender']}

Total encounters: {selected_patient['total_encounters']}
Total conditions: {selected_patient['total_conditions']}
Distinct condition types: {selected_patient['distinct_condition_types']}

Total observations: {selected_patient['total_observations']}
Distinct observation types: {selected_patient['distinct_observation_types']}

Total procedures: {selected_patient['total_procedures']}
Distinct procedure types: {selected_patient['distinct_procedure_types']}

Total medication requests: {selected_patient['total_medication_requests']}
Distinct medications: {selected_patient['distinct_medications']}

Total clinical events: {selected_patient['total_clinical_events']}
"""

# ------------------------------------------------------------
# Retrieve RAG context again from the persisted RAG corpus
# ------------------------------------------------------------

rag_context_df = (
    rag_corpus_df
    .filter(F.col("document_type") == "patient_profile")
    .filter(
        F.col("document_id") ==
        selected_patient["patient_id"]
    )
    .select("document_text")
)

rag_context_rows = rag_context_df.collect()

if rag_context_rows:
    retrieved_context = "\n".join(
        row["document_text"]
        for row in rag_context_rows
    )
else:
    retrieved_context = "No additional RAG context available."

# ------------------------------------------------------------
# Build grounded healthcare insight prompt
# ------------------------------------------------------------

healthcare_insight_prompt = f"""
You are a healthcare data analytics assistant.

Analyze the following structured Patient 360 information.

PATIENT PROFILE:
{patient_profile_text}

RETRIEVED RAG CONTEXT:
{retrieved_context}

Instructions:
1. Use only the provided patient profile and retrieved context.
2. Summarize recorded healthcare activity.
3. Highlight notable patterns in encounters, conditions,
   observations, procedures, and medications.
4. Do not diagnose any disease.
5. Do not recommend treatment or medication changes.
6. Do not infer information that is not present.
7. Clearly distinguish recorded facts from analytical observations.
8. If information is unavailable, explicitly state that.

Provide a concise healthcare analytics summary.
"""

print("=" * 60)
print("HEALTHCARE INSIGHT PROMPT")
print("=" * 60)

print(f"Patient selected: {selected_patient['patient_id']}")
print(f"RAG context available: {len(retrieved_context) > 0}")

print("\nPrompt successfully created.")

# COMMAND ----------

# ============================================================
# CELL 9 — HEALTHCARE INSIGHT VALIDATION
# ============================================================

validation_checks = [
    (
        "patient_profile_available",
        patient_profile_text,
        len(patient_profile_text) > 0
    ),
    (
        "rag_context_available",
        retrieved_context,
        len(retrieved_context) > 0
    ),
    (
        "grounded_prompt_available",
        healthcare_insight_prompt,
        len(healthcare_insight_prompt) > 0
    )
]

insight_validation_df = spark.createDataFrame(
    [
        (check, len(str(value)), "PASS" if passed else "FAIL")
        for check, value, passed in validation_checks
    ],
    ["validation_check", "content_length", "status"]
)

display(insight_validation_df)

# COMMAND ----------

# ============================================================
# CELL 10 — FINAL HEALTHCARE INSIGHT OUTPUT
# ============================================================

final_insight_df = (
    patient_insight_df
    .withColumn(
        "insight_summary",
        F.concat_ws(
            " ",
            F.lit("Patient has"),
            F.col("total_encounters"),
            F.lit("recorded encounters,"),
            F.col("total_conditions"),
            F.lit("recorded conditions,"),
            F.col("total_observations"),
            F.lit("observations,"),
            F.col("total_procedures"),
            F.lit("procedures, and"),
            F.col("total_medication_requests"),
            F.lit("medication requests."),
            F.lit("The recorded clinical activity level is"),
            F.when(
                F.col("total_clinical_events") == 0,
                F.lit("none.")
            )
            .when(
                F.col("total_clinical_events") <= 20,
                F.lit("low.")
            )
            .when(
                F.col("total_clinical_events") <= 100,
                F.lit("moderate.")
            )
            .otherwise(
                F.lit("high.")
            )
        )
    )
    .select(
        "patient_id",
        "age",
        "gender",
        "total_encounters",
        "total_conditions",
        "total_observations",
        "total_procedures",
        "total_medication_requests",
        "total_clinical_events",
        "insight_summary"
    )
)

print("=" * 60)
print("FINAL AI HEALTHCARE INSIGHTS")
print("=" * 60)

display(
    final_insight_df
    .orderBy(F.desc("total_clinical_events"))
    .limit(10)
)

print(
    f"Healthcare insight records: "
    f"{final_insight_df.count():,}"
)

print("\n✓ 08_AI_Healthcare_Insights COMPLETED")

# COMMAND ----------

