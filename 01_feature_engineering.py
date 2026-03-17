# Databricks notebook source
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
    .select(
        "observation_time",
        "temperature_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "relative_humidity",
        "windspeed",
        "year", "month", "day", "hour", "weekday"
    ) \
    .orderBy("observation_time")

total = df.count()
print(f"Registros carregados: {total:,}")
print(f"   Período: {df.agg(F.min('observation_time')).collect()[0][0]} → "
      f"{df.agg(F.max('observation_time')).collect()[0][0]}")

# COMMAND ----------

# Criar features de lag e rolling window
print("Criando features...\n")

# Window ordenada por tempo - para features de lag
w = Window.orderBy("observation_time")

PI = math.pi

df_features = (
    df

    # Target
    # Chuva na PRÓXIMA hora (shift -1)
    .withColumn("target_rain_next_1h",
        F.lead("precipitation", 1).over(w))
    .withColumn("target_will_rain",
        F.when(F.lead("precipitation", 1).over(w) > 0, 1)
         .otherwise(0))

    # Lag features
    .withColumn("temp_lag_1h",
        F.lag("temperature_2m", 1).over(w))
    .withColumn("temp_lag_3h",
        F.lag("temperature_2m", 3).over(w))
    .withColumn("temp_lag_24h",
        F.lag("temperature_2m", 24).over(w))

    .withColumn("precip_lag_1h",
        F.lag("precipitation", 1).over(w))
    .withColumn("precip_lag_3h",
        F.lag("precipitation", 3).over(w))
    .withColumn("precip_lag_24h",
        F.lag("precipitation", 24).over(w))

    .withColumn("humidity_lag_1h",
        F.lag("relative_humidity", 1).over(w))
    .withColumn("humidity_lag_3h",
        F.lag("relative_humidity", 3).over(w))

    # Delta features (variação) 
    .withColumn("temp_delta_1h",
        F.col("temperature_2m") - F.lag("temperature_2m", 1).over(w))
    .withColumn("humidity_delta_1h",
        F.col("relative_humidity") - F.lag("relative_humidity", 1).over(w))
    .withColumn("precip_delta_1h",
        F.col("precipitation") - F.lag("precipitation", 1).over(w))

    # Rolling window features
    .withColumn("temp_rolling_6h",
        F.round(F.avg("temperature_2m").over(
            w.rowsBetween(-5, 0)), 3))
    .withColumn("humidity_rolling_6h",
        F.round(F.avg("relative_humidity").over(
            w.rowsBetween(-5, 0)), 3))
    .withColumn("precip_rolling_3h",
        F.round(F.sum("precipitation").over(
            w.rowsBetween(-2, 0)), 3))
    .withColumn("precip_rolling_6h",
        F.round(F.sum("precipitation").over(
            w.rowsBetween(-5, 0)), 3))
    .withColumn("precip_rolling_24h",
        F.round(F.sum("precipitation").over(
            w.rowsBetween(-23, 0)), 3))

    # Encoding cíclico
    .withColumn("hour_sin",
        F.round(F.sin(F.col("hour") * (2 * PI / 24)), 6))
    .withColumn("hour_cos",
        F.round(F.cos(F.col("hour") * (2 * PI / 24)), 6))
    .withColumn("month_sin",
        F.round(F.sin(F.col("month") * (2 * PI / 12)), 6))
    .withColumn("month_cos",
        F.round(F.cos(F.col("month") * (2 * PI / 12)), 6))

    # Flags sazonais
    .withColumn("is_rainy_season",
        F.when(F.col("month").isin(10, 11, 12, 1, 2, 3), 1)
         .otherwise(0))
    .withColumn("is_peak_rain_month",
        F.when(F.col("month").isin(1, 2, 12), 1)
         .otherwise(0))
    .withColumn("is_afternoon",
        F.when(F.col("hour").between(12, 18), 1)
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

ML_FEATURES = [
    # Meteorológicas atuais
    "temperature_2m", "apparent_temperature",
    "precipitation", "relative_humidity", "windspeed",

    # Lag
    "temp_lag_1h", "temp_lag_3h", "temp_lag_24h",
    "precip_lag_1h", "precip_lag_3h", "precip_lag_24h",
    "humidity_lag_1h", "humidity_lag_3h",

    # Delta
    "temp_delta_1h", "humidity_delta_1h", "precip_delta_1h",

    # Rolling
    "temp_rolling_6h", "humidity_rolling_6h",
    "precip_rolling_3h", "precip_rolling_6h", "precip_rolling_24h",

    # Temporais cíclicas
    "hour_sin", "hour_cos", "month_sin", "month_cos",

    # Temporais brutas
    "hour", "month", "day", "year", "weekday",

    # Flags
    "is_rainy_season", "is_peak_rain_month", "is_afternoon",
]

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