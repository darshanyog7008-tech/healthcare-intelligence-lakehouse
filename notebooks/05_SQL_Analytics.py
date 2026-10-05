# Databricks notebook source
# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 1 — GOLD TABLE LOAD & BASIC VALIDATION
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_patients,
# MAGIC     COUNT(DISTINCT patient_id) AS distinct_patients,
# MAGIC     COUNT(*) - COUNT(DISTINCT patient_id) AS duplicate_patient_rows
# MAGIC FROM healthcare_lakehouse.gold.patient_360;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 2 — PATIENT DEMOGRAPHICS
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT
# MAGIC     gender,
# MAGIC     COUNT(*) AS patient_count,
# MAGIC     ROUND(
# MAGIC         COUNT(*) * 100.0 /
# MAGIC         SUM(COUNT(*)) OVER (),
# MAGIC         2
# MAGIC     ) AS percentage_of_patients
# MAGIC FROM healthcare_lakehouse.gold.patient_360
# MAGIC GROUP BY gender
# MAGIC ORDER BY patient_count DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 3 — AGE GROUP ANALYSIS
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT
# MAGIC     CASE
# MAGIC         WHEN age < 18 THEN '0-17'
# MAGIC         WHEN age BETWEEN 18 AND 29 THEN '18-29'
# MAGIC         WHEN age BETWEEN 30 AND 44 THEN '30-44'
# MAGIC         WHEN age BETWEEN 45 AND 59 THEN '45-59'
# MAGIC         ELSE '60+'
# MAGIC     END AS age_group,
# MAGIC
# MAGIC     COUNT(*) AS patient_count,
# MAGIC
# MAGIC     ROUND(
# MAGIC         COUNT(*) * 100.0 /
# MAGIC         SUM(COUNT(*)) OVER (),
# MAGIC         2
# MAGIC     ) AS percentage_of_patients
# MAGIC
# MAGIC FROM healthcare_lakehouse.gold.patient_360
# MAGIC
# MAGIC GROUP BY
# MAGIC     CASE
# MAGIC         WHEN age < 18 THEN '0-17'
# MAGIC         WHEN age BETWEEN 18 AND 29 THEN '18-29'
# MAGIC         WHEN age BETWEEN 30 AND 44 THEN '30-44'
# MAGIC         WHEN age BETWEEN 45 AND 59 THEN '45-59'
# MAGIC         ELSE '60+'
# MAGIC     END
# MAGIC
# MAGIC ORDER BY
# MAGIC     CASE age_group
# MAGIC         WHEN '0-17' THEN 1
# MAGIC         WHEN '18-29' THEN 2
# MAGIC         WHEN '30-44' THEN 3
# MAGIC         WHEN '45-59' THEN 4
# MAGIC         WHEN '60+' THEN 5
# MAGIC     END;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 4 — ENCOUNTER UTILIZATION
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT
# MAGIC     patient_id,
# MAGIC     age,
# MAGIC     gender,
# MAGIC     total_encounters,
# MAGIC
# MAGIC     CASE
# MAGIC         WHEN total_encounters >= 50 THEN 'HIGH'
# MAGIC         WHEN total_encounters >= 20 THEN 'MEDIUM'
# MAGIC         ELSE 'LOW'
# MAGIC     END AS encounter_utilization,
# MAGIC
# MAGIC     first_encounter_time,
# MAGIC     last_encounter_time
# MAGIC
# MAGIC FROM healthcare_lakehouse.gold.patient_360
# MAGIC
# MAGIC WHERE total_encounters > 0
# MAGIC
# MAGIC ORDER BY total_encounters DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 5 — TOP PATIENTS BY CLINICAL ACTIVITY
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT
# MAGIC     patient_id,
# MAGIC     age,
# MAGIC     gender,
# MAGIC     total_clinical_events,
# MAGIC
# MAGIC     RANK() OVER (
# MAGIC         ORDER BY total_clinical_events DESC
# MAGIC     ) AS clinical_activity_rank
# MAGIC
# MAGIC FROM healthcare_lakehouse.gold.patient_360
# MAGIC
# MAGIC ORDER BY clinical_activity_rank
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 6 — MOST RECENT PATIENT ACTIVITY
# MAGIC -- ============================================================
# MAGIC
# MAGIC WITH ranked_patients AS (
# MAGIC
# MAGIC     SELECT
# MAGIC         patient_id,
# MAGIC         age,
# MAGIC         gender,
# MAGIC         total_encounters,
# MAGIC         last_encounter_time,
# MAGIC
# MAGIC         ROW_NUMBER() OVER (
# MAGIC             ORDER BY last_encounter_time DESC
# MAGIC         ) AS row_num
# MAGIC
# MAGIC     FROM healthcare_lakehouse.gold.patient_360
# MAGIC
# MAGIC     WHERE last_encounter_time IS NOT NULL
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     patient_id,
# MAGIC     age,
# MAGIC     gender,
# MAGIC     total_encounters,
# MAGIC     last_encounter_time,
# MAGIC     row_num
# MAGIC
# MAGIC FROM ranked_patients
# MAGIC
# MAGIC WHERE row_num <= 20
# MAGIC
# MAGIC ORDER BY row_num;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 7 — CONDITION BURDEN ANALYSIS
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT
# MAGIC     patient_id,
# MAGIC     age,
# MAGIC     gender,
# MAGIC     total_conditions,
# MAGIC     distinct_condition_types,
# MAGIC
# MAGIC     CASE
# MAGIC         WHEN distinct_condition_types >= 20 THEN 'HIGH'
# MAGIC         WHEN distinct_condition_types >= 10 THEN 'MEDIUM'
# MAGIC         ELSE 'LOW'
# MAGIC     END AS condition_burden_category
# MAGIC
# MAGIC FROM healthcare_lakehouse.gold.patient_360
# MAGIC
# MAGIC WHERE total_conditions > 0
# MAGIC
# MAGIC ORDER BY distinct_condition_types DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 8 — OBSERVATION INTENSITY ANALYSIS
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT
# MAGIC     patient_id,
# MAGIC     age,
# MAGIC     gender,
# MAGIC     total_encounters,
# MAGIC     total_observations,
# MAGIC     observations_per_encounter,
# MAGIC
# MAGIC     CASE
# MAGIC         WHEN observations_per_encounter >= 10 THEN 'HIGH'
# MAGIC         WHEN observations_per_encounter >= 5 THEN 'MEDIUM'
# MAGIC         ELSE 'LOW'
# MAGIC     END AS observation_intensity
# MAGIC
# MAGIC FROM healthcare_lakehouse.gold.patient_360
# MAGIC
# MAGIC WHERE total_encounters > 0
# MAGIC
# MAGIC ORDER BY observations_per_encounter DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 9 — PROCEDURE UTILIZATION
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT
# MAGIC     patient_id,
# MAGIC     age,
# MAGIC     gender,
# MAGIC     total_encounters,
# MAGIC     total_procedures,
# MAGIC     distinct_procedure_types,
# MAGIC     procedures_per_encounter,
# MAGIC
# MAGIC     CASE
# MAGIC         WHEN procedures_per_encounter >= 5 THEN 'HIGH'
# MAGIC         WHEN procedures_per_encounter >= 2 THEN 'MEDIUM'
# MAGIC         ELSE 'LOW'
# MAGIC     END AS procedure_utilization
# MAGIC
# MAGIC FROM healthcare_lakehouse.gold.patient_360
# MAGIC
# MAGIC WHERE total_encounters > 0
# MAGIC
# MAGIC ORDER BY procedures_per_encounter DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 10 — MEDICATION UTILIZATION
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT
# MAGIC     patient_id,
# MAGIC     age,
# MAGIC     gender,
# MAGIC     total_encounters,
# MAGIC     total_medication_requests,
# MAGIC     distinct_medications,
# MAGIC     medications_per_encounter,
# MAGIC
# MAGIC     CASE
# MAGIC         WHEN medications_per_encounter >= 3 THEN 'HIGH'
# MAGIC         WHEN medications_per_encounter >= 1 THEN 'MEDIUM'
# MAGIC         ELSE 'LOW'
# MAGIC     END AS medication_utilization
# MAGIC
# MAGIC FROM healthcare_lakehouse.gold.patient_360
# MAGIC
# MAGIC WHERE total_encounters > 0
# MAGIC
# MAGIC ORDER BY medications_per_encounter DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 11 — PATIENT CLINICAL PROFILE USING CTE
# MAGIC -- ============================================================
# MAGIC
# MAGIC WITH patient_profile AS (
# MAGIC
# MAGIC     SELECT
# MAGIC         patient_id,
# MAGIC         age,
# MAGIC         gender,
# MAGIC         total_encounters,
# MAGIC         total_conditions,
# MAGIC         total_observations,
# MAGIC         total_procedures,
# MAGIC         total_medication_requests,
# MAGIC
# MAGIC         (
# MAGIC             total_conditions +
# MAGIC             total_observations +
# MAGIC             total_procedures +
# MAGIC             total_medication_requests
# MAGIC         ) AS total_recorded_events
# MAGIC
# MAGIC     FROM healthcare_lakehouse.gold.patient_360
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     patient_id,
# MAGIC     age,
# MAGIC     gender,
# MAGIC     total_encounters,
# MAGIC     total_conditions,
# MAGIC     total_observations,
# MAGIC     total_procedures,
# MAGIC     total_medication_requests,
# MAGIC     total_recorded_events
# MAGIC
# MAGIC FROM patient_profile
# MAGIC
# MAGIC ORDER BY total_recorded_events DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 12 — RANK PATIENTS WITHIN AGE GROUPS
# MAGIC -- ============================================================
# MAGIC
# MAGIC WITH patient_groups AS (
# MAGIC
# MAGIC     SELECT
# MAGIC         patient_id,
# MAGIC         age,
# MAGIC         gender,
# MAGIC         total_clinical_events,
# MAGIC
# MAGIC         CASE
# MAGIC             WHEN age < 18 THEN '0-17'
# MAGIC             WHEN age BETWEEN 18 AND 29 THEN '18-29'
# MAGIC             WHEN age BETWEEN 30 AND 44 THEN '30-44'
# MAGIC             WHEN age BETWEEN 45 AND 59 THEN '45-59'
# MAGIC             ELSE '60+'
# MAGIC         END AS age_group
# MAGIC
# MAGIC     FROM healthcare_lakehouse.gold.patient_360
# MAGIC ),
# MAGIC
# MAGIC ranked_patients AS (
# MAGIC
# MAGIC     SELECT
# MAGIC         patient_id,
# MAGIC         age,
# MAGIC         gender,
# MAGIC         age_group,
# MAGIC         total_clinical_events,
# MAGIC
# MAGIC         RANK() OVER (
# MAGIC             PARTITION BY age_group
# MAGIC             ORDER BY total_clinical_events DESC
# MAGIC         ) AS activity_rank
# MAGIC
# MAGIC     FROM patient_groups
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     patient_id,
# MAGIC     age,
# MAGIC     gender,
# MAGIC     age_group,
# MAGIC     total_clinical_events,
# MAGIC     activity_rank
# MAGIC
# MAGIC FROM ranked_patients
# MAGIC
# MAGIC WHERE activity_rank <= 3
# MAGIC
# MAGIC ORDER BY
# MAGIC     age_group,
# MAGIC     activity_rank;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 13 — CONDITIONAL AGGREGATION
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_patients,
# MAGIC
# MAGIC     SUM(
# MAGIC         CASE
# MAGIC             WHEN total_encounters >= 50 THEN 1
# MAGIC             ELSE 0
# MAGIC         END
# MAGIC     ) AS high_encounter_patients,
# MAGIC
# MAGIC     SUM(
# MAGIC         CASE
# MAGIC             WHEN total_conditions >= 30 THEN 1
# MAGIC             ELSE 0
# MAGIC         END
# MAGIC     ) AS high_condition_patients,
# MAGIC
# MAGIC     SUM(
# MAGIC         CASE
# MAGIC             WHEN total_observations >= 200 THEN 1
# MAGIC             ELSE 0
# MAGIC         END
# MAGIC     ) AS high_observation_patients,
# MAGIC
# MAGIC     SUM(
# MAGIC         CASE
# MAGIC             WHEN total_procedures >= 150 THEN 1
# MAGIC             ELSE 0
# MAGIC         END
# MAGIC     ) AS high_procedure_patients,
# MAGIC
# MAGIC     SUM(
# MAGIC         CASE
# MAGIC             WHEN total_medication_requests >= 50 THEN 1
# MAGIC             ELSE 0
# MAGIC         END
# MAGIC     ) AS high_medication_patients
# MAGIC
# MAGIC FROM healthcare_lakehouse.gold.patient_360;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 14 — MOST RECENT PATIENT ACTIVITY BY AGE GROUP
# MAGIC -- ============================================================
# MAGIC
# MAGIC WITH patient_groups AS (
# MAGIC
# MAGIC     SELECT
# MAGIC         patient_id,
# MAGIC         age,
# MAGIC         gender,
# MAGIC         last_encounter_time,
# MAGIC
# MAGIC         CASE
# MAGIC             WHEN age < 18 THEN '0-17'
# MAGIC             WHEN age BETWEEN 18 AND 29 THEN '18-29'
# MAGIC             WHEN age BETWEEN 30 AND 44 THEN '30-44'
# MAGIC             WHEN age BETWEEN 45 AND 59 THEN '45-59'
# MAGIC             ELSE '60+'
# MAGIC         END AS age_group
# MAGIC
# MAGIC     FROM healthcare_lakehouse.gold.patient_360
# MAGIC
# MAGIC     WHERE last_encounter_time IS NOT NULL
# MAGIC ),
# MAGIC
# MAGIC ranked_activity AS (
# MAGIC
# MAGIC     SELECT
# MAGIC         patient_id,
# MAGIC         age,
# MAGIC         gender,
# MAGIC         age_group,
# MAGIC         last_encounter_time,
# MAGIC
# MAGIC         ROW_NUMBER() OVER (
# MAGIC             PARTITION BY age_group
# MAGIC             ORDER BY last_encounter_time DESC
# MAGIC         ) AS rn
# MAGIC
# MAGIC     FROM patient_groups
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     patient_id,
# MAGIC     age,
# MAGIC     gender,
# MAGIC     age_group,
# MAGIC     last_encounter_time
# MAGIC
# MAGIC FROM ranked_activity
# MAGIC
# MAGIC WHERE rn = 1
# MAGIC
# MAGIC ORDER BY age_group;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 15 — MULTI-METRIC PATIENT ANALYSIS
# MAGIC -- ============================================================
# MAGIC
# MAGIC WITH patient_metrics AS (
# MAGIC
# MAGIC     SELECT
# MAGIC         patient_id,
# MAGIC         age,
# MAGIC         gender,
# MAGIC         total_encounters,
# MAGIC         total_conditions,
# MAGIC         total_observations,
# MAGIC         total_procedures,
# MAGIC         total_medication_requests,
# MAGIC
# MAGIC         (
# MAGIC             total_encounters +
# MAGIC             total_conditions +
# MAGIC             total_observations +
# MAGIC             total_procedures +
# MAGIC             total_medication_requests
# MAGIC         ) AS combined_activity
# MAGIC
# MAGIC     FROM healthcare_lakehouse.gold.patient_360
# MAGIC ),
# MAGIC
# MAGIC ranked_patients AS (
# MAGIC
# MAGIC     SELECT
# MAGIC         *,
# MAGIC         RANK() OVER (
# MAGIC             ORDER BY combined_activity DESC
# MAGIC         ) AS activity_rank
# MAGIC
# MAGIC     FROM patient_metrics
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     patient_id,
# MAGIC     age,
# MAGIC     gender,
# MAGIC     total_encounters,
# MAGIC     total_conditions,
# MAGIC     total_observations,
# MAGIC     total_procedures,
# MAGIC     total_medication_requests,
# MAGIC     combined_activity,
# MAGIC     activity_rank
# MAGIC
# MAGIC FROM ranked_patients
# MAGIC
# MAGIC WHERE activity_rank <= 10
# MAGIC
# MAGIC ORDER BY activity_rank;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ============================================================
# MAGIC -- CELL 16 — FINAL SQL DATA QUALITY VALIDATION
# MAGIC -- ============================================================
# MAGIC
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_rows,
# MAGIC
# MAGIC     COUNT(DISTINCT patient_id) AS distinct_patients,
# MAGIC
# MAGIC     SUM(
# MAGIC         CASE
# MAGIC             WHEN patient_id IS NULL
# MAGIC                  OR TRIM(patient_id) = ''
# MAGIC             THEN 1
# MAGIC             ELSE 0
# MAGIC         END
# MAGIC     ) AS null_or_empty_patient_ids,
# MAGIC
# MAGIC     SUM(
# MAGIC         CASE
# MAGIC             WHEN total_encounters < 0
# MAGIC               OR total_conditions < 0
# MAGIC               OR total_observations < 0
# MAGIC               OR total_procedures < 0
# MAGIC               OR total_medication_requests < 0
# MAGIC             THEN 1
# MAGIC             ELSE 0
# MAGIC         END
# MAGIC     ) AS invalid_metric_rows
# MAGIC
# MAGIC FROM healthcare_lakehouse.gold.patient_360;

# COMMAND ----------

