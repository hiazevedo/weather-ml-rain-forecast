# Databricks notebook source
# Configurações
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import os
from pyspark.sql import functions as F
from datetime import datetime, timezone
import pytz

# Workaround obrigatório no Databricks Free Edition Serverless
mlflow.autolog(disable=True)

# Serverless (runtime 19.x): sem URIs explícitos o MLflow tenta ler a config
# spark.mlflow.modelRegistryUri, bloqueada no Spark Connect (CONFIG_NOT_AVAILABLE)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
MLFLOW_TMP = "/Volumes/weather_pipeline/bronze/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP

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

# Carregar modelo mais recente do experimento MLflow (sem Model Registry)
print("Carregando modelo do MLflow...\n")

CURRENT_USER    = spark.sql("SELECT current_user()").collect()[0][0]
EXPERIMENT_NAME = f"/Users/{CURRENT_USER}/weather-ml-rain-forecast"

# Sem fallback: se o modelo não puder ser carregado, o notebook falha.
# (Antes, qualquer erro aqui treinava em silêncio um RF reduzido — 100 árvores,
# profundidade 10, 2018-2022, 50% de amostra — e publicava previsões piores
# sem nenhum aviso.)
try:
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="params.type = 'classifier'",
        order_by=["start_time DESC"],
    )
except Exception as e:
    raise RuntimeError(
        f"Falha ao consultar o experimento MLflow '{EXPERIMENT_NAME}': "
        f"{type(e).__name__}: {e}"
    ) from e

if runs.empty:
    raise RuntimeError(
        f"Nenhum run de classificador (params.type='classifier') encontrado em "
        f"'{EXPERIMENT_NAME}'. Execute 03_model_training ou 06_retrain_model antes."
    )

run_id = runs.iloc[0]["run_id"]


def load_classifier(run_id: str):
    """Carrega o modelo de um run, nos dois formatos de armazenamento do MLflow.

    - MLflow 2.x (Serverless env <= 2): artefato do run -> runs:/<run_id>/model
    - MLflow 3.x (Serverless env >= 5): o modelo vira um LoggedModel
      (models:/m-...) e NÃO existe em runs:/<run_id>/model (404).
    """
    client = mlflow.tracking.MlflowClient()
    exp_id = runs.iloc[0]["experiment_id"]

    # 1) MLflow 3: LoggedModel gerado por este run
    if hasattr(client, "search_logged_models"):
        logged = [m for m in client.search_logged_models(experiment_ids=[exp_id])
                  if getattr(m, "source_run_id", None) == run_id]
        if logged:
            logged.sort(key=lambda m: m.creation_timestamp, reverse=True)
            uri = f"models:/{logged[0].model_id}"
            return mlflow.sklearn.load_model(uri), uri

    # 2) MLflow 2: artefato do run
    uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(uri), uri


try:
    model, model_uri = load_classifier(run_id)
except Exception as e:
    raise RuntimeError(
        f"Falha ao carregar o modelo do run {run_id}: {type(e).__name__}: {e}"
    ) from e

print(f"   Modelo carregado do MLflow | {model_uri} | run_id: {run_id} "
      f"({runs.iloc[0].get('tags.mlflow.runName')})")

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
            F.current_timestamp() + F.expr("INTERVAL 24 HOURS")) \
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