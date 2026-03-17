# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql import Window
import math

# COMMAND ----------

# Atualizar gold.rain_features com dados novos
print("Atualizando gold.rain_features...\n")

PI = math.pi
FEATURE_TABLE  = "weather_pipeline.gold.rain_features"
SILVER_TABLE   = "weather_pipeline.silver.weather_clean"

# Verificar última data na feature store
ultima_data = spark.sql(f"""
    SELECT MAX(observation_time) AS ultima
    FROM {FEATURE_TABLE}
""").collect()[0][0]

print(f"   Última data na feature store : {ultima_data}")

# Buscar apenas registros novos da Silver
df_novos = spark.table(SILVER_TABLE) \
    .filter(F.col("is_forecast") == False) \
    .filter(F.col("observation_time") > ultima_data) \
    .filter(F.col("precipitation").isNotNull()) \
    .filter(F.col("temperature_2m").isNotNull()) \
    .filter(F.col("relative_humidity").isNotNull()) \
    .select(
        "observation_time", "temperature_2m",
        "apparent_temperature", "precipitation",
        "rain", "relative_humidity", "windspeed",
        "year", "month", "day", "hour", "weekday"
    )

novos_count = df_novos.count()
print(f"   Novos registros encontrados  : {novos_count:,}")

if novos_count == 0:
    print("\n Nenhum dado novo — feature store já atualizada!")
    dbutils.notebook.exit("UP_TO_DATE")

# COMMAND ----------t

# Calcular features com contexto histórico
print("Calculando features...\n")

# Pegar últimas 24h da feature store para calcular lags corretamente
# Pegar contexto da feature store
df_contexto = spark.table(FEATURE_TABLE) \
    .filter(F.col("observation_time") > F.date_sub(
        F.lit(ultima_data), 2)) \
    .select(
        "observation_time", "temperature_2m",
        "apparent_temperature", "precipitation",
        "relative_humidity", "windspeed",
        "year", "month", "day", "hour", "weekday"
    )

# df_novos já vem da Silver com year e day — garantir mesmo select
df_novos_clean = df_novos.select(
    "observation_time", "temperature_2m",
    "apparent_temperature", "precipitation",
    "relative_humidity", "windspeed",
    "year", "month", "day", "hour", "weekday"
)

df_combined = df_contexto.unionByName(df_novos_clean) \
    .orderBy("observation_time")

w = Window.orderBy("observation_time")

df_features = (
    df_combined
    .withColumn("target_rain_next_1h", F.lead("precipitation", 1).over(w))
    .withColumn("target_will_rain",    F.when(F.lead("precipitation", 1).over(w) > 0, 1).otherwise(0))
    .withColumn("temp_lag_1h",         F.lag("temperature_2m", 1).over(w))
    .withColumn("temp_lag_3h",         F.lag("temperature_2m", 3).over(w))
    .withColumn("temp_lag_24h",        F.lag("temperature_2m", 24).over(w))
    .withColumn("precip_lag_1h",       F.lag("precipitation", 1).over(w))
    .withColumn("precip_lag_3h",       F.lag("precipitation", 3).over(w))
    .withColumn("precip_lag_24h",      F.lag("precipitation", 24).over(w))
    .withColumn("humidity_lag_1h",     F.lag("relative_humidity", 1).over(w))
    .withColumn("humidity_lag_3h",     F.lag("relative_humidity", 3).over(w))
    .withColumn("temp_delta_1h",       F.col("temperature_2m") - F.lag("temperature_2m", 1).over(w))
    .withColumn("humidity_delta_1h",   F.col("relative_humidity") - F.lag("relative_humidity", 1).over(w))
    .withColumn("precip_delta_1h",     F.col("precipitation") - F.lag("precipitation", 1).over(w))
    .withColumn("temp_rolling_6h",     F.round(F.avg("temperature_2m").over(w.rowsBetween(-5, 0)), 3))
    .withColumn("humidity_rolling_6h", F.round(F.avg("relative_humidity").over(w.rowsBetween(-5, 0)), 3))
    .withColumn("precip_rolling_3h",   F.round(F.sum("precipitation").over(w.rowsBetween(-2, 0)), 3))
    .withColumn("precip_rolling_6h",   F.round(F.sum("precipitation").over(w.rowsBetween(-5, 0)), 3))
    .withColumn("precip_rolling_24h",  F.round(F.sum("precipitation").over(w.rowsBetween(-23, 0)), 3))
    .withColumn("hour_sin",            F.round(F.sin(F.col("hour") * (2 * math.pi / 24)), 6))
    .withColumn("hour_cos",            F.round(F.cos(F.col("hour") * (2 * math.pi / 24)), 6))
    .withColumn("month_sin",           F.round(F.sin(F.col("month") * (2 * math.pi / 12)), 6))
    .withColumn("month_cos",           F.round(F.cos(F.col("month") * (2 * math.pi / 12)), 6))
    .withColumn("is_rainy_season",     F.when(F.col("month").isin(10,11,12,1,2,3), 1).otherwise(0))
    .withColumn("is_peak_rain_month",  F.when(F.col("month").isin(1,2,12), 1).otherwise(0))
    .withColumn("is_afternoon",        F.when(F.col("hour").between(12, 18), 1).otherwise(0))
    # Filtrar apenas os registros novos (remover contexto)
    .filter(F.col("observation_time") > ultima_data)
    .filter(F.col("temp_lag_24h").isNotNull())
    .filter(F.col("target_rain_next_1h").isNotNull())
)

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

# Select final — observation_time separado para evitar duplicata
df_final = df_features.select(
    ["observation_time"] +
    ML_FEATURES +
    ["target_will_rain", "target_rain_next_1h"]
)

novos_features = df_final.count()
print(f"✅ Features calculadas: {novos_features:,} novos registros")

# COMMAND ----------

# Append na feature store
print("Salvando novos registros na feature store...\n")

# APPEND — não sobrescreve dados históricos
df_final.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable(FEATURE_TABLE)

total = spark.table(FEATURE_TABLE).count()
nova_ultima = spark.sql(f"""
    SELECT MAX(observation_time) AS ultima
    FROM {FEATURE_TABLE}
""").collect()[0][0]

print(f"""
FEATURE STORE ATUALIZADA
 - Novos registros adicionados : {novos_features:<10,}
 - Total na feature store      : {total:<10,}
 - Última observação           : {str(nova_ultima)[:19]}
""")