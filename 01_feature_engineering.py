# Databricks notebook source
# MAGIC %run ./features

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import IntegerType

SILVER_TABLE  = "weather_pipeline.silver.weather_clean"
FEATURE_TABLE = "weather_pipeline.gold.rain_features"

print("Configurações carregadas")
print(f"   Fonte   : {SILVER_TABLE}")
print(f"   Destino : {FEATURE_TABLE}")

# COMMAND ----------

# Carregar dados históricos
import math

print("Carregando dados históricos...\n")

df = spark.table(SILVER_TABLE) \
    .filter(F.col("is_forecast") == False) \
    .filter(F.col("precipitation").isNotNull()) \
    .filter(F.col("temperature_2m").isNotNull()) \
    .filter(F.col("relative_humidity").isNotNull()) \
    .select(*SILVER_COLS) \
    .orderBy("observation_time")

total = df.count()
print(f"Registros carregados: {total:,}")
print(f"   Período: {df.agg(F.min('observation_time')).collect()[0][0]} → "
      f"{df.agg(F.max('observation_time')).collect()[0][0]}")

# COMMAND ----------

# Criar features de lag e rolling window
print("Criando features...\n")

# Window particionada por ano para distribuir a carga entre executors
# Trade-off: lags que cruzam virada de ano (Jan 1, 00h-23h) ficam nulos,
# mas esses registros são removidos pelo filter de temp_lag_24h.isNotNull() abaixo
w = Window.partitionBy("year").orderBy("observation_time")

PI = math.pi

df_features = (
    add_features(df, w)

    # Target
    # Chuva na PRÓXIMA hora (shift -1)
    .withColumn("target_rain_next_1h",
        F.lead("precipitation", 1).over(w))
    .withColumn("target_will_rain",
        F.when(F.lead("precipitation", 1).over(w) > 0, 1)
         .otherwise(0))

    # Remover nulos críticos (primeiras 24h sem lag)
    .filter(F.col("temp_lag_24h").isNotNull())
    .filter(F.col("precip_lag_24h").isNotNull())
    .filter(F.col("target_rain_next_1h").isNotNull())
)

total = df_features.count()
print(f"   Features criadas: {total:,} registros")
print(f"   Features totais : {len(df_features.columns)} colunas")

# Distribuição do target
print("\n Distribuição do target:")
display(df_features.groupBy("target_will_rain") \
    .count() \
    .withColumn("pct", F.round(
        F.col("count") * 100.0 / total, 1)) \
    .orderBy("target_will_rain"))

# COMMAND ----------

# Salvar Feature Store
print("Salvando Feature Store...\n")

# ML_FEATURES vem de ./features (variante B: base + extras)

# Select final — observation_time fora do ML_FEATURES para evitar duplicata
df_final = df_features.select(
    ["observation_time"] +
    ML_FEATURES +
    ["target_will_rain", "target_rain_next_1h"]
)

df_final.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(FEATURE_TABLE)

print(f"   Feature Store salva: {FEATURE_TABLE}")
print(f"   Registros : {df_final.count():,}")
print(f"   Features  : {len(ML_FEATURES)}")

# Qualidade das features
print("\n Qualidade — nulos por feature:")
null_counts = [(c, df_final.filter(F.col(c).isNull()).count())
               for c in ML_FEATURES]
for feat, nulls in sorted(null_counts, key=lambda x: x[1], reverse=True)[:10]:
    if nulls > 0:
        pct = nulls / df_final.count() * 100
        print(f"      {feat:<30} {nulls:>8,} nulos ({pct:.1f}%)")

print("\n FEATURE ENGINEERING CONCLUÍDO!")