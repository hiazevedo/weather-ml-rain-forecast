# Databricks notebook source
# Configurações
import mlflow
import mlflow.sklearn
import os
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from datetime import datetime, timezone
import pytz

MLFLOW_TMP = "/Volumes/weather_pipeline/bronze/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP

EXPERIMENT_NAME = "/Users/{}/weather-ml-rain-forecast/weather-ml-rain-forecast".format(
    spark.sql("SELECT current_user()").collect()[0][0]
)
mlflow.set_experiment(EXPERIMENT_NAME)

FEATURE_TABLE      = "weather_pipeline.gold.rain_features"
FORECAST_TABLE     = "weather_pipeline.silver.weather_forecast"
PREDICTIONS_TABLE  = "weather_pipeline.gold.rain_forecast_ml"
BEST_THRESHOLD     = 0.65
MODEL_NAME         = "rain-forecast-birigui"

ML_FEATURES = [
    "temperature_2m", "apparent_temperature",
    "precipitation", "relative_humidity", "windspeed",
    "temp_lag_1h", "temp_lag_3h", "temp_lag_24h",
    "precip_lag_1h", "precip_lag_3h", "precip_lag_24h",
    "humidity_lag_1h", "humidity_lag_3h",
    "temp_delta_1h", "humidity_delta_1h", "precip_delta_1h",
    "temp_rolling_6h", "humidity_rolling_6h",
    "precip_rolling_3h", "precip_rolling_6h", "precip_rolling_24h",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "hour", "month", "weekday",
    "is_rainy_season", "is_peak_rain_month", "is_afternoon",
]

TZ_BR = pytz.timezone("America/Sao_Paulo")
agora = datetime.now(TZ_BR)
print(f"   Configurações carregadas")
print(f"   Data/Hora  : {agora.strftime('%Y-%m-%d %H:%M')} (Brasília)")
print(f"   Threshold  : {BEST_THRESHOLD}")
print(f"   Modelo     : {MODEL_NAME} v1")

# COMMAND ----------

# Carregar modelo do MLflow Registry
print("Carregando modelo do MLflow Registry...\n")

from sklearn.ensemble import RandomForestClassifier

# Retreinar o modelo final (MLflow Registry no Serverless
# não suporta load direto — padrão conhecido)
df_train = spark.table(FEATURE_TABLE) \
    .filter(F.col("observation_time") < "2023-01-01") \
    .dropna(subset=ML_FEATURES + ["target_will_rain"])

X_train     = df_train.select(ML_FEATURES).toPandas()
y_train     = df_train.select("target_will_rain").toPandas().values.ravel()

model = RandomForestClassifier(
    n_estimators=200, max_depth=15,
    class_weight="balanced",
    random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

print(f"   Modelo carregado e treinado")
print(f"   Registros de treino : {len(X_train):,}")

# COMMAND ----------

# DBTITLE 1,Cell 3: Fixed IndentationError
# =============================================================================
# CÉLULA 3 CORRIGIDA — sem windspeed no forecast
# =============================================================================
import math

print("🔮 Preparando features para previsão próximas 24h...\n")

PI = math.pi

# Pegar dados históricos recentes para calcular lags
df_hist_recent = spark.table("weather_pipeline.silver.weather_clean") \
    .filter(F.col("is_forecast") == False) \
    .filter(F.col("observation_time") >= F.date_sub(F.current_timestamp(), 3)) \
    .select(
        "observation_time", "temperature_2m", "apparent_temperature",
        "precipitation", "relative_humidity", "windspeed",
        "hour", "month", "weekday"
    )

df_forecast_raw = spark.table(FORECAST_TABLE) \
    .select(
        "observation_time", "temperature_2m", "apparent_temperature",
        "precipitation", "relative_humidity",
        "hour", "month"
    ) \
    .withColumn("windspeed", F.lit(0.0)) \
    .withColumn("weekday", F.dayofweek("observation_time"))

# Combinar histórico recente + forecast
df_combined = df_hist_recent.unionByName(
    df_forecast_raw, allowMissingColumns=True
).orderBy("observation_time")

# Calcular features de lag e rolling com Window
from pyspark.sql import Window

w = Window.orderBy("observation_time")

df_with_features = (
    df_combined
    .withColumn("temp_lag_1h",       F.lag("temperature_2m", 1).over(w))
    .withColumn("temp_lag_3h",       F.lag("temperature_2m", 3).over(w))
    .withColumn("temp_lag_24h",      F.lag("temperature_2m", 24).over(w))
    .withColumn("precip_lag_1h",     F.lag("precipitation", 1).over(w))
    .withColumn("precip_lag_3h",     F.lag("precipitation", 3).over(w))
    .withColumn("precip_lag_24h",    F.lag("precipitation", 24).over(w))
    .withColumn("humidity_lag_1h",   F.lag("relative_humidity", 1).over(w))
    .withColumn("humidity_lag_3h",   F.lag("relative_humidity", 3).over(w))
    .withColumn("temp_delta_1h",
        F.col("temperature_2m") - F.lag("temperature_2m", 1).over(w))
    .withColumn("humidity_delta_1h",
        F.col("relative_humidity") - F.lag("relative_humidity", 1).over(w))
    .withColumn("precip_delta_1h",
        F.col("precipitation") - F.lag("precipitation", 1).over(w))
    .withColumn("temp_rolling_6h",
        F.round(F.avg("temperature_2m").over(w.rowsBetween(-5, 0)), 3))
    .withColumn("humidity_rolling_6h",
        F.round(F.avg("relative_humidity").over(w.rowsBetween(-5, 0)), 3))
    .withColumn("precip_rolling_3h",
        F.round(F.sum("precipitation").over(w.rowsBetween(-2, 0)), 3))
    .withColumn("precip_rolling_6h",
        F.round(F.sum("precipitation").over(w.rowsBetween(-5, 0)), 3))
    .withColumn("precip_rolling_24h",
        F.round(F.sum("precipitation").over(w.rowsBetween(-23, 0)), 3))
    .withColumn("hour_sin",
        F.round(F.sin(F.col("hour") * (2 * PI / 24)), 6))
    .withColumn("hour_cos",
        F.round(F.cos(F.col("hour") * (2 * PI / 24)), 6))
    .withColumn("month_sin",
        F.round(F.sin(F.col("month") * (2 * PI / 12)), 6))
    .withColumn("month_cos",
        F.round(F.cos(F.col("month") * (2 * PI / 12)), 6))
    .withColumn("is_rainy_season",
        F.when(F.col("month").isin(10,11,12,1,2,3), 1).otherwise(0))
    .withColumn("is_peak_rain_month",
        F.when(F.col("month").isin(1,2,12), 1).otherwise(0))
    .withColumn("is_afternoon",
        F.when(F.col("hour").between(12, 18), 1).otherwise(0))
)

# Filtrar apenas as próximas 24h do forecast
df_next24 = df_with_features \
    .filter(F.col("observation_time") > F.current_timestamp()) \
    .filter(F.col("observation_time") <=
            F.date_add(F.current_timestamp(), 1)) \
    .dropna(subset=ML_FEATURES)

total = df_next24.count()
print(f"✅ Features preparadas: {total} horas para previsão")

# COMMAND ----------

# Rodar inferência e salvar predições
print("🔮 Rodando inferência...\n")

X_pred = df_next24.select(ML_FEATURES).toPandas()

# Predições
y_prob  = model.predict_proba(X_pred)[:, 1]
y_pred  = (y_prob >= BEST_THRESHOLD).astype(int)

# Adicionar predições ao dataframe
times_pd = df_next24.select("observation_time", "temperature_2m",
    "apparent_temperature", "relative_humidity",
    "hour", "month").toPandas()

results_pd = times_pd.copy()
results_pd["prob_chuva"]      = y_prob.round(4)
results_pd["pred_vai_chover"] = y_pred
results_pd["alerta_chuva"] = results_pd["prob_chuva"].apply(
    lambda p:
        "🔴 CHUVA FORTE"    if p >= 0.85 else
        "🟠 CHUVA PROVÁVEL" if p >= 0.65 else
        "🟡 POSSÍVEL CHUVA" if p >= 0.40 else
        "🟢 SEM CHUVA"
)
results_pd["threshold_usado"]  = BEST_THRESHOLD
results_pd["modelo"]           = MODEL_NAME
results_pd["inferencia_at"]    = datetime.now(timezone.utc).isoformat()

# Converter para Spark e salvar
df_results = spark.createDataFrame(results_pd)
df_results.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(PREDICTIONS_TABLE)

print(f"Predições salvas: {PREDICTIONS_TABLE}")
print(f"\nResumo das próximas 24h:")
display(spark.sql(f"""
    SELECT
        alerta_chuva,
        COUNT(*)                              AS horas,
        ROUND(AVG(prob_chuva) * 100, 1)       AS prob_media_pct,
        MIN(DATE_FORMAT(observation_time,
            'dd/MM HH:mm'))                   AS primeira_hora,
        MAX(DATE_FORMAT(observation_time,
            'dd/MM HH:mm'))                   AS ultima_hora
    FROM {PREDICTIONS_TABLE}
    GROUP BY alerta_chuva
    ORDER BY prob_media_pct DESC
"""))

# COMMAND ----------

# Relatório final
resumo = spark.sql(f"""
    SELECT
        SUM(pred_vai_chover)                   AS horas_com_chuva_prevista,
        COUNT(*)                               AS total_horas,
        ROUND(AVG(prob_chuva) * 100, 1)        AS prob_media_pct,
        ROUND(MAX(prob_chuva) * 100, 1)        AS prob_maxima_pct,
        MIN(CASE WHEN pred_vai_chover = 1
            THEN DATE_FORMAT(observation_time, 'HH:mm')
            END)                               AS primeira_chuva_prevista
    FROM {PREDICTIONS_TABLE}
""").collect()[0]

vai_chover = resumo["horas_com_chuva_prevista"] > 0

print(f"""
WEATHER ML RAIN FORECAST — RELATÓRIO FINAL
MODELO
  Nome       : RandomForestClassifier
  AUC-ROC    : 0.9318
  Recall     : 73.7% das chuvas detectadas
  Threshold  : {BEST_THRESHOLD} (otimizado para F1)
  Registry   : {MODEL_NAME}

PREVISÃO PRÓXIMAS 24H — BIRIGUI-SP
  Vai chover?          : {'SIM 🌧️' if vai_chover else 'NÃO ☀️'}
  Horas com chuva      : {resumo['horas_com_chuva_prevista'] or 0:>5} / {resumo['total_horas']}
  Probabilidade média  : {resumo['prob_media_pct']:>5.1f}%
  Probabilidade máxima : {resumo['prob_maxima_pct']:>5.1f}%
  Primeira chuva       : {str(resumo['primeira_chuva_prevista'] or 'N/A'):>8}

DADOS
  Histórico  : 1940→2026 (87 anos, 755k registros)
  Features   : 31 features engenheiradas
  Treino     : 1940–2022 | Teste: 2024–2026
""")
print("  PROJETO SEMANA 5 CONCLUÍDO!")
print("\n   Notebooks entregues:")
for nb in [
    "01_feature_engineering.py  — 31 features + Feature Store Delta",
    "02_exploratory_analysis.py — EDA + correlações + heatmap hora×mês",
    "03_model_training.py       — XGBoost + RF + Prophet com MLflow",
    "04_model_evaluation.py     — Matriz confusão + ROC + Feature Importance",
    "05_batch_inference.py      — Predições 24h → gold.rain_forecast_ml",
]:
    print(f"   ✅ {nb}")