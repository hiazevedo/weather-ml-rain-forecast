# Databricks notebook source
# MOS multimodelo — produção: chance de chuva MEDIDA em Birigui, hora a hora,
# nas próximas 48 h, combinando as previsões atuais de 6 modelos.
#
# Modelo: workspace.default.rain-forecast-mos-multimodelo@champion (16_train_mos_multimodelo);
# limiar na tag "threshold" da versão.
# Ele foi treinado com as rodadas de 1 dia antes; nas primeiras horas a
# antecedência é menor, e o desempenho medido (experimento 14) vale para ~1 dia.
#
# Saídas:
#   gold.rain_forecast_mos          — a previsão mais recente (painel)
#   gold.rain_forecast_mos_history  — todas as previsões, para medir o acerto depois
#   Volume bronze.forecast_multimodelo — a resposta crua da Open-Meteo (arquivo próprio)
# Job: [weather-ml] MOS multimodelo (a cada 3 h).

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.1"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./mos

# COMMAND ----------

import json
from datetime import datetime, timedelta, timezone
import pandas as pd
import pytz
import requests
import mlflow
import mlflow.sklearn

mlflow.autolog(disable=True)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

LAT, LON = -21.2878, -50.3409
MODEL_NAME = "workspace.default.rain-forecast-mos-multimodelo"
PRED_TABLE = "weather_pipeline.gold.rain_forecast_mos"
HIST_TABLE = "weather_pipeline.gold.rain_forecast_mos_history"
RAW = "/Volumes/weather_pipeline/bronze/forecast_multimodelo"
HORIZONTE_H = 48
spark.sql("CREATE VOLUME IF NOT EXISTS weather_pipeline.bronze.forecast_multimodelo")

TZ = pytz.timezone("America/Sao_Paulo")
agora = datetime.now(TZ).replace(tzinfo=None)
hora_cheia = agora.replace(minute=0, second=0, microsecond=0)

# COMMAND ----------

# Previsão atual dos 6 modelos (past_days=1 dá a hora anterior para a janela t−1..t+1)
r = requests.get("https://api.open-meteo.com/v1/forecast", params={
    "latitude": LAT, "longitude": LON, "hourly": ",".join(MOS_VARS), "models": ",".join(MOS_MODELOS),
    "past_days": 1, "forecast_days": 3, "timezone": "America/Sao_Paulo"}, timeout=60)
r.raise_for_status()
resposta = r.json()
dbutils.fs.put(f"{RAW}/{agora:%Y/%m}/mos_{agora:%Y%m%d_%H%M%S}.json",
               json.dumps({"coletado_em": datetime.now(timezone.utc).isoformat(), **resposta}), overwrite=True)

s = pd.DataFrame(resposta["hourly"])
s["observation_time"] = pd.to_datetime(s.pop("time"))
s = s.set_index("observation_time").sort_index()
X = features_mos(s, coluna_forecast)
alvo = X[(X.index > hora_cheia) & (X.index <= hora_cheia + timedelta(hours=HORIZONTE_H))]
alvo = alvo[alvo["p_best_match"].notna()]

# COMMAND ----------

mv = mlflow.tracking.MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
versao, limiar = mv.version, float(mv.tags["threshold"])
modelo = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
prob = modelo.predict_proba(alvo[list(modelo.feature_names_in_)])[:, 1]

res = pd.DataFrame({
    "target_time": alvo.index,
    "antecedencia_h": ((alvo.index - hora_cheia) / pd.Timedelta(hours=1)).astype(int),
    "prob_chuva_medida": prob.round(4),
    "pred_chuva_medida": (prob >= limiar).astype(int),
    "threshold": limiar,
    "modelos_com_chuva": (alvo["ens_frac"] * len(MOS_MODELOS)).round(),
    "chuva_media_modelos_mm": alvo["ens_media"].round(2),
    "chuva_api_mm": alvo["p_best_match"].round(2),
    "modelo": f"mos-multimodelo v{versao}",
    "inferencia_local": pd.Timestamp(agora),
})
df = spark.createDataFrame(res)
df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PRED_TABLE)

df.createOrReplaceTempView("_mos_novas")
spark.sql(f"CREATE TABLE IF NOT EXISTS {HIST_TABLE} AS SELECT * FROM _mos_novas WHERE 1 = 0")
spark.sql(f"""
    MERGE INTO {HIST_TABLE} h USING _mos_novas n
      ON h.inferencia_local = n.inferencia_local AND h.target_time = n.target_time
    WHEN NOT MATCHED THEN INSERT *
""")

amanha = (agora + timedelta(days=1)).date()
r_am = res[res["target_time"].dt.date == amanha]
print(f"mos-multimodelo v{versao} (limiar {limiar}) | {len(res)} horas | "
      f"amanhã: {int(r_am['pred_chuva_medida'].sum())} h com chuva prevista, "
      f"máx. {r_am['prob_chuva_medida'].max():.0%}")
