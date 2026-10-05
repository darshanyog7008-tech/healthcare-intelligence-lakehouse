# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Bronze Layer
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC The Bronze layer stores the raw healthcare FHIR resources after ingestion from the source volume.
# MAGIC
# MAGIC The goal of this layer is to preserve the source data with minimal transformation while making the data available for downstream processing.
# MAGIC
# MAGIC ### Bronze Layer Principles
# MAGIC
# MAGIC - Preserve source information
# MAGIC - Perform minimal transformations
# MAGIC - Maintain the original FHIR resource structure
# MAGIC - Capture all relevant clinical resource types
# MAGIC - Provide a reliable foundation for the Silver layer
# MAGIC
# MAGIC ### Source
# MAGIC
# MAGIC Catalog: `healthcare_lakehouse`
# MAGIC
# MAGIC Schema: `default`
# MAGIC
# MAGIC Volume: `healthcare_source`
# MAGIC
# MAGIC Source path:
# MAGIC
# MAGIC `/Volumes/healthcare_lakehouse/default/healthcare_source/`

# COMMAND ----------

# Cell 2 — Import required libraries

from pyspark.sql import functions as F

# COMMAND ----------

# Cell 3 — Define source path

source_path = "/Volumes/healthcare_lakehouse/default/healthcare_source/"

print(f"Source path: {source_path}")

# COMMAND ----------

# Cell 4 — Inspect source files

source_files = dbutils.fs.ls(source_path)

print(f"Total files found: {len(source_files)}")

for file_info in source_files[:10]:
    print(file_info.name)

# COMMAND ----------

# Cell 5 — Read FHIR JSON bundles

raw_fhir_df = (
    spark.read
    .option("multiline", "true")
    .json(source_path)
)

print(f"FHIR bundles loaded: {raw_fhir_df.count():,}")

print("\nTop-level schema:")
raw_fhir_df.printSchema()

# COMMAND ----------

# Cell 6 — Explode FHIR Bundle entries

bronze_entries_df = (
    raw_fhir_df
    .select(
        F.explode("entry").alias("entry")
    )
)

print(f"Total FHIR entries: {bronze_entries_df.count():,}")

bronze_entries_df.printSchema()

# COMMAND ----------

# Cell 7 — Extract FHIR resource fields

bronze_resources_df = (
    bronze_entries_df
    .select("entry.resource.*")
)

print(f"Total Bronze resources: {bronze_resources_df.count():,}")

print("\nResource types:")
(
    bronze_resources_df
    .groupBy("resourceType")
    .count()
    .orderBy(F.desc("count"))
    .show(50, truncate=False)
)

# COMMAND ----------

# Cell 8 — Validate Bronze resources

print("=== BRONZE RESOURCE VALIDATION ===")

# Total records
total_records = bronze_resources_df.count()
print(f"Total records: {total_records:,}")

# Null resource types
null_resource_types = (
    bronze_resources_df
    .filter(F.col("resourceType").isNull())
    .count()
)

print(f"Null resourceType: {null_resource_types:,}")

# Distinct resource types
distinct_resource_types = (
    bronze_resources_df
    .select("resourceType")
    .distinct()
    .count()
)

print(f"Distinct resource types: {distinct_resource_types:,}")

# Null resource IDs
null_resource_ids = (
    bronze_resources_df
    .filter(F.col("id").isNull())
    .count()
)

print(f"Null resource IDs: {null_resource_ids:,}")

# COMMAND ----------

# Cell 9 — Bronze resource distribution

resource_distribution = (
    bronze_resources_df
    .groupBy("resourceType")
    .count()
    .orderBy(F.desc("count"))
)

resource_distribution.show(50, truncate=False)

# COMMAND ----------

# Cell 10 — Bronze validation summary

total_records = bronze_resources_df.count()

distinct_resources = (
    bronze_resources_df
    .select("resourceType")
    .distinct()
    .count()
)

null_resource_type = (
    bronze_resources_df
    .filter(F.col("resourceType").isNull())
    .count()
)

null_resource_id = (
    bronze_resources_df
    .filter(F.col("id").isNull())
    .count()
)

print("========================================")
print("       BRONZE LAYER VALIDATION")
print("========================================")
print(f"Total Records          : {total_records:,}")
print(f"Resource Types         : {distinct_resources}")
print(f"Null Resource Types    : {null_resource_type:,}")
print(f"Null Resource IDs      : {null_resource_id:,}")
print("========================================")

# COMMAND ----------

# Cell 11 — Check duplicate resource IDs

duplicate_ids = (
    bronze_resources_df
    .groupBy("resourceType", "id")
    .count()
    .filter(F.col("count") > 1)
)

duplicate_count = duplicate_ids.count()

print(f"Duplicate resource keys: {duplicate_count:,}")

if duplicate_count > 0:
    print("\nSample duplicates:")
    duplicate_ids.show(20, truncate=False)
else:
    print("No duplicate resourceType + id combinations found.")

# COMMAND ----------

# Cell 12 — Create Bronze schema

spark.sql("""
CREATE SCHEMA IF NOT EXISTS healthcare_lakehouse.bronze
""")

print("Bronze schema is ready.")

# COMMAND ----------

# Cell 13 — Persist Bronze resources

bronze_table = "healthcare_lakehouse.bronze.fhir_resources"

(
    bronze_resources_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(bronze_table)
)

print(f"Bronze table created: {bronze_table}")

# COMMAND ----------

# Cell 14 — Verify Bronze table

bronze_df = spark.table("healthcare_lakehouse.bronze.fhir_resources")

print(f"Bronze table records: {bronze_df.count():,}")

bronze_df.printSchema()

# COMMAND ----------

