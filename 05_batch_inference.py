# Databricks notebook source
# MAGIC %run ./features

# COMMAND ----------

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
HISTORY_TABLE      = "weather_pipeline.gold.rain_forecast_ml_history"
ACCURACY_VIEW      = "weather_pipeline.gold.rain_model_accuracy_hourly"
MODEL_NAME         = "rain-forecast-birigui"
UC_MODEL_NAME      = "workspace.default.rain-forecast-birigui"

# As features vêm de ./features; as colunas que o modelo espera vêm do
# próprio modelo (feature_names_in_), então a inferência acompanha o champion.

TZ_BR = pytz.timezone("America/Sao_Paulo")
agora = datetime.now(TZ_BR)
print(f"   Configurações carregadas")
print(f"   Data/Hora  : {agora.strftime('%Y-%m-%d %H:%M')} (Brasília)")
print(f"   Modelo     : {UC_MODEL_NAME}@champion")

# COMMAND ----------

# Carregar o modelo "champion" do Model Registry (Unity Catalog).
# O alias é promovido pelo 06_retrain_model apenas quando o F1 não piora.
# Sem fallback: se o modelo não puder ser carregado, o notebook falha.
# (Antes, qualquer erro aqui treinava em silêncio um RF reduzido — 100 árvores,
# profundidade 10, 2018-2022, 50% de amostra — e publicava previsões piores
# sem nenhum aviso.)
print("Carregando modelo champion do Model Registry...\n")

client = mlflow.tracking.MlflowClient()
try:
    mv_champion = client.get_model_version_by_alias(UC_MODEL_NAME, "champion")
    model = mlflow.sklearn.load_model(f"models:/{UC_MODEL_NAME}@champion")
except Exception as e:
    raise RuntimeError(
        f"Falha ao carregar '{UC_MODEL_NAME}@champion': {type(e).__name__}: {e}. "
        f"Execute 06_retrain_model para registrar/promover uma versão."
    ) from e

MODEL_VERSION = mv_champion.version
# Limiar de decisão escolhido pelo 06_retrain_model e gravado na versão do modelo
if "threshold" not in mv_champion.tags:
    raise RuntimeError(
        f"{UC_MODEL_NAME} v{MODEL_VERSION} não tem a tag 'threshold'. "
        f"Retreine com 06_retrain_model ou defina a tag na versão.")
BEST_THRESHOLD = float(mv_champion.tags["threshold"])
print(f"   Limiar     : {BEST_THRESHOLD} (tag da versão)")
print(f"   Modelo carregado | {UC_MODEL_NAME} v{MODEL_VERSION} (@champion) "
      f"| run_id: {mv_champion.run_id}")

# COMMAND ----------

# DBTITLE 1,Cell 3: Fixed IndentationError
# =============================================================================
# Features das próximas horas: histórico recente observado + previsão
# =============================================================================
import math

print("🔮 Preparando features para previsão próximas 24h...\n")

PI = math.pi

# Pegar dados históricos recentes para calcular lags
df_hist_recent = spark.table("weather_pipeline.silver.weather_clean") \
    .filter(F.col("is_forecast") == False) \
    .filter(F.col("observation_time") >= F.date_sub(F.current_timestamp(), 3)) \
    .select(*SILVER_COLS)

# windspeed vem da previsão (antes era fixado em 0.0, mas o modelo foi
# treinado com vento real — todas as previsões futuras viam "vento zero")
df_forecast_raw = spark.table(FORECAST_TABLE) \
    .select(*[c for c in SILVER_COLS if c != "weekday"]) \
    .withColumn("weekday", F.dayofweek("observation_time"))

# Combinar histórico recente + forecast
df_combined = df_hist_recent.unionByName(
    df_forecast_raw, allowMissingColumns=True
).orderBy("observation_time")

# Calcular features de lag e rolling com Window
from pyspark.sql import Window

w = Window.orderBy("observation_time")

df_with_features = (
    add_features(df_combined, w)
    # Chuva que a API prevê para a hora-alvo (t+1): baseline para comparar com o modelo
    .withColumn("api_precip_alvo", F.lead("precipitation", 1).over(w))
)

MODEL_FEATURES = list(model.feature_names_in_)
print(f"   Modelo usa {len(MODEL_FEATURES)} features")

# Filtrar apenas as próximas 24h do forecast.
# observation_time é hora LOCAL sem fuso: comparar com current_timestamp() (UTC)
# deslocava a janela em 3h. Usamos a hora local (naive) no mesmo formato.
# A linha da hora cheia atual (t0, já observada) prevê t0+1 — a próxima hora.
agora_local = datetime.now(TZ_BR).strftime("%Y-%m-%d %H:%M:%S")
t0 = F.lit(datetime.now(TZ_BR).strftime("%Y-%m-%d %H:00:00")).cast("timestamp")
df_next24 = df_with_features \
    .filter(F.col("observation_time") >= t0) \
    .filter(F.col("observation_time") <  t0 + F.expr("INTERVAL 24 HOURS")) \
    .dropna(subset=MODEL_FEATURES)

total = df_next24.count()
print(f"✅ Features preparadas: {total} horas para previsão")

# COMMAND ----------

# Rodar inferência e salvar predições
print("🔮 Rodando inferência...\n")

X_pred = df_next24.select(MODEL_FEATURES).toPandas()

# Predições
y_prob  = model.predict_proba(X_pred)[:, 1]
y_pred  = (y_prob >= BEST_THRESHOLD).astype(int)

# Adicionar predições ao dataframe
times_pd = df_next24.select("observation_time", "temperature_2m",
    "apparent_temperature", "relative_humidity",
    "hour", "month", "api_precip_alvo").toPandas()
times_pd["api_precip_alvo"] = times_pd["api_precip_alvo"].astype("float64")

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
# O modelo recebe as features da hora t e prevê chuva na hora t+1
# (target_will_rain = precipitação de t+1 > 0). target_time é a hora prevista.
results_pd["target_time"]      = results_pd["observation_time"] + pd.Timedelta(hours=1)
results_pd["inferencia_local"] = pd.Timestamp(agora_local)  # hora local, comparável a target_time
results_pd["threshold_usado"]  = BEST_THRESHOLD
results_pd["modelo"]           = f"{MODEL_NAME} v{MODEL_VERSION}"
results_pd["inferencia_at"]    = datetime.now(timezone.utc).isoformat()

# Converter para Spark e salvar
df_results = spark.createDataFrame(results_pd)
df_results.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(PREDICTIONS_TABLE)

print(f"Predições salvas: {PREDICTIONS_TABLE}")

# Histórico (append): permite medir acurácia REAL comparando cada previsão
# com a chuva observada depois. Idempotente por (inferencia_local, target_time).
df_results.createOrReplaceTempView("_novas_predicoes")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} AS
    SELECT * FROM _novas_predicoes WHERE 1 = 0
""")
if "api_precip_alvo" not in spark.table(HISTORY_TABLE).columns:
    spark.sql(f"ALTER TABLE {HISTORY_TABLE} ADD COLUMNS (api_precip_alvo DOUBLE)")
spark.sql(f"""
    MERGE INTO {HISTORY_TABLE} h
    USING _novas_predicoes n
      ON h.inferencia_local = n.inferencia_local AND h.target_time = n.target_time
    WHEN NOT MATCHED THEN INSERT *
""")

# Acurácia real por hora: última previsão feita ANTES da hora-alvo vs chuva
# observada (silver.weather_clean com is_forecast = false).
spark.sql(f"""
    CREATE OR REPLACE VIEW {ACCURACY_VIEW} AS
    WITH ultima AS (
        SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY target_time ORDER BY inferencia_local DESC) AS rn
        FROM {HISTORY_TABLE}
        WHERE inferencia_local < target_time
    )
    SELECT
        u.target_time,
        u.inferencia_local,
        ROUND(u.prob_chuva, 3)                       AS prob_chuva,
        u.pred_vai_chover,
        u.threshold_usado,
        u.modelo,
        u.api_precip_alvo,
        CAST(u.api_precip_alvo > 0 AS INT)           AS api_vai_chover,
        o.precipitation                              AS chuva_real_mm,
        CAST(o.precipitation > 0 AS INT)             AS choveu,
        CASE
            WHEN o.precipitation > 0  AND u.pred_vai_chover = 1 THEN 'Verdadeiro Positivo'
            WHEN o.precipitation = 0  AND u.pred_vai_chover = 0 THEN 'Verdadeiro Negativo'
            WHEN o.precipitation > 0  AND u.pred_vai_chover = 0 THEN 'Falso Negativo'
            WHEN o.precipitation = 0  AND u.pred_vai_chover = 1 THEN 'Falso Positivo'
        END                                          AS resultado_classificacao
    FROM ultima u
    JOIN weather_pipeline.silver.weather_clean o
      ON o.observation_time = u.target_time AND o.is_forecast = false
    WHERE u.rn = 1
""")
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
  Threshold  : {BEST_THRESHOLD} (tag da versão — máx. F1 na calibração)
  Registry   : {UC_MODEL_NAME} v{MODEL_VERSION}

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