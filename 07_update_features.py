# Databricks notebook source
# MAGIC %run ./features

# COMMAND ----------

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

# Fonte única: Silver (já deduplicada por observation_time). Pegamos 48h de
# contexto ANTES do ponto de corte + os registros novos, tudo da mesma tabela,
# para que os lags/rolling (calculados por posição de linha) sejam consistentes.
df_novos = spark.table(SILVER_TABLE) \
    .filter(F.col("is_forecast") == False) \
    .filter(F.col("observation_time") > ultima_data) \
    .filter(F.col("precipitation").isNotNull()) \
    .filter(F.col("temperature_2m").isNotNull()) \
    .filter(F.col("relative_humidity").isNotNull()) \
    .dropDuplicates(["observation_time"])

novos_count = df_novos.count()
print(f"   Novos registros encontrados  : {novos_count:,}")

if novos_count == 0:
    print("\n Nenhum dado novo — feature store já atualizada!")
    dbutils.notebook.exit("UP_TO_DATE")

# COMMAND ----------

# Calcular features com contexto histórico
print("Calculando features...\n")

COLS = SILVER_COLS   # de ./features

df_contexto = spark.table(SILVER_TABLE) \
    .filter(F.col("is_forecast") == False) \
    .filter(F.col("observation_time") <= ultima_data) \
    .filter(F.col("observation_time") > F.expr(f"timestamp'{ultima_data}' - INTERVAL 48 HOURS")) \
    .filter(F.col("precipitation").isNotNull()) \
    .filter(F.col("temperature_2m").isNotNull()) \
    .filter(F.col("relative_humidity").isNotNull()) \
    .dropDuplicates(["observation_time"]) \
    .select(*COLS)

df_novos_clean = df_novos.select(*COLS)

df_combined = df_contexto.unionByName(df_novos_clean) \
    .orderBy("observation_time")

w = Window.orderBy("observation_time")

df_features = (
    add_features(df_combined, w)
    .withColumn("target_rain_next_1h", F.lead("precipitation", 1).over(w))
    .withColumn("target_will_rain",    F.when(F.lead("precipitation", 1).over(w) > 0, 1).otherwise(0))
    # Filtrar apenas os registros novos (remover contexto)
    .filter(F.col("observation_time") > ultima_data)
    .filter(F.col("temp_lag_24h").isNotNull())
    .filter(F.col("precip_lag_24h").isNotNull())
    .filter(F.col("target_rain_next_1h").isNotNull())
)

# ML_FEATURES vem de ./features (variante B: base + extras)

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
    .option("mergeSchema", "true") \
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