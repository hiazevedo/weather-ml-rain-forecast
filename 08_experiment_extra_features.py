# Databricks notebook source
# Experimento: variáveis extras do ERA5 e o índice ENSO melhoram o modelo?
#
# Variantes (mesmas linhas, mesmo split temporal e mesma regra de limiar do
# 06_retrain_model — só muda o conjunto de features):
#   A  base            — as 31 features atuais
#   B  base + ERA5     — ponto de orvalho, pressão e tendência, nebulosidade por
#                        camada, direção do vento, vapor d'água na coluna, camada limite
#   C  base + ERA5 + ENSO
#   D  base + ENSO     — isola o efeito do ENSO
#
# As variáveis extras existem também na API de previsão, então um modelo B/C
# pode ser usado em produção. Nada aqui altera o champion.

# COMMAND ----------

import json, time
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd
import requests
import mlflow
from pyspark.sql import functions as F, Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                             precision_score, recall_score)

mlflow.autolog(disable=True)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/{}/weather-ml-rain-forecast".format(
    spark.sql("SELECT current_user()").collect()[0][0]))

LAT, LON, TZ = -21.2878, -50.3409, "America/Sao_Paulo"
EXTRA_VARS = ["dew_point_2m", "pressure_msl", "cloud_cover", "cloud_cover_low",
              "cloud_cover_mid", "cloud_cover_high", "wind_direction_10m",
              "total_column_integrated_water_vapour", "boundary_layer_height"]
spark.sql("CREATE VOLUME IF NOT EXISTS weather_pipeline.bronze.era5_extra")
RAW = "/Volumes/weather_pipeline/bronze/era5_extra"
EXTRA_TABLE = "weather_pipeline.silver.era5_extra_hourly"
RESULTS_TABLE = "weather_pipeline.gold.ml_experiments"

BASE = [
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
ERA5 = ["dew_point_2m", "dewpoint_depression", "pressure_msl", "pressure_tend_3h",
        "pressure_tend_24h", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
        "cloud_cover_high", "cloud_low_delta_1h", "wind_dir_sin", "wind_dir_cos",
        "tcwv", "tcwv_delta_3h", "boundary_layer_height"]
ENSO = ["oni"]

# COMMAND ----------

# 1) Coleta das variáveis extras (Archive/ERA5), um arquivo por ano.
#    Anos já baixados são reaproveitados; o ano corrente é sempre atualizado.
fim = date.today() - timedelta(days=5)
existentes = {f.name for f in dbutils.fs.ls(RAW)}

def fetch(start, end, tentativas=6):
    for i in range(tentativas):
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": LAT, "longitude": LON, "start_date": start, "end_date": end,
            "hourly": ",".join(EXTRA_VARS), "timezone": TZ}, timeout=180)
        if r.status_code == 429:          # limite da API gratuita: espera e tenta de novo
            time.sleep(60 * (i + 1)); continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Archive recusou {start}..{end} após {tentativas} tentativas")

for ano in range(1940, fim.year + 1):
    nome = f"era5_extra_{ano}.json"
    if nome in existentes and ano < fim.year:
        continue
    d = fetch(f"{ano}-01-01", min(date(ano, 12, 31), fim).isoformat())
    dbutils.fs.put(f"{RAW}/{nome}", json.dumps(d["hourly"]), overwrite=True)
    time.sleep(3)
print("Coleta ok:", len(dbutils.fs.ls(RAW)), "arquivos")

# COMMAND ----------

# 2) Tabela horária das variáveis extras
frames = []
for f in dbutils.fs.ls(RAW):
    h = json.loads(dbutils.fs.head(f.path, 50_000_000))
    frames.append(pd.DataFrame(h))
extra = pd.concat(frames, ignore_index=True).drop_duplicates("time")
(spark.createDataFrame(extra)
    .withColumn("observation_time", F.to_timestamp("time"))
    .drop("time")
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(EXTRA_TABLE))
print(f"{EXTRA_TABLE}: {len(extra):,} horas")

# COMMAND ----------

# 3) Features derivadas (janelas por posição de linha, como no 01 — a série é contínua)
w = Window.orderBy("observation_time")
ex = (spark.table(EXTRA_TABLE)
    .withColumn("pressure_tend_3h",  F.col("pressure_msl") - F.lag("pressure_msl", 3).over(w))
    .withColumn("pressure_tend_24h", F.col("pressure_msl") - F.lag("pressure_msl", 24).over(w))
    .withColumn("cloud_low_delta_1h", F.col("cloud_cover_low") - F.lag("cloud_cover_low", 1).over(w))
    .withColumn("tcwv", F.col("total_column_integrated_water_vapour"))
    .withColumn("tcwv_delta_3h", F.col("tcwv") - F.lag("tcwv", 3).over(w))
    .withColumn("wind_dir_sin", F.sin(F.radians("wind_direction_10m")))
    .withColumn("wind_dir_cos", F.cos(F.radians("wind_direction_10m"))))

# ENSO: ONI disponível no mês (available_month), com o último valor publicado
# propagado para meses ainda sem índice novo — nunca usa um valor do futuro.
oni = spark.table("weather_pipeline.silver.enso_oni").select(
    F.col("available_month").alias("mes"), "oni")
meses = spark.sql("SELECT explode(sequence(DATE'1950-01-01', trunc(current_date(), 'MM'), INTERVAL 1 MONTH)) AS mes")
oni_mes = (meses.join(oni, "mes", "left")
    .withColumn("oni", F.last("oni", ignorenulls=True).over(
        Window.orderBy("mes").rowsBetween(Window.unboundedPreceding, 0))))

df = (spark.table("weather_pipeline.gold.rain_features")
    .join(ex, "observation_time", "inner")
    .withColumn("dewpoint_depression", F.col("temperature_2m") - F.col("dew_point_2m"))
    .withColumn("mes", F.trunc("observation_time", "MM"))
    .join(oni_mes, "mes", "left")
    .filter(F.col("observation_time") < F.date_sub(F.current_date(), 1))
    .dropna(subset=BASE + ERA5 + ENSO + ["target_will_rain"]))   # mesmas linhas em todas as variantes

# 4) Split idêntico ao 06_retrain_model
cols = ["observation_time"] + BASE + ERA5 + ENSO + ["target_will_rain"]
pdf = df.select(*cols).toPandas()
t = pdf["observation_time"]
c24 = pd.Timestamp(date.today()) - pd.DateOffset(months=24)
c12 = pd.Timestamp(date.today()) - pd.DateOffset(months=12)
train, cal, test = pdf[t < c24], pdf[(t >= c24) & (t < c12)], pdf[t >= c12]
print(f"linhas: treino {len(train):,} | calibração {len(cal):,} | teste {len(test):,} "
      f"(de {t.min()} a {t.max()})")

# COMMAND ----------

VARIANTES = {"A_base": BASE, "B_base_era5": BASE + ERA5,
             "C_base_era5_enso": BASE + ERA5 + ENSO, "D_base_enso": BASE + ENSO}
y_tr, y_cal, y_te = (x["target_will_rain"].values for x in (train, cal, test))
resultados, importancias = [], {}

for nome, feats in VARIANTES.items():
    m = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced",
                               random_state=42, n_jobs=-1)
    m.fit(train[feats], y_tr)
    p_cal = m.predict_proba(cal[feats])[:, 1]
    thr = float(max(np.round(np.arange(0.30, 0.91, 0.05), 2),
                    key=lambda th: f1_score(y_cal, (p_cal >= th).astype(int))))
    p = m.predict_proba(test[feats])[:, 1]
    yp = (p >= thr).astype(int)
    r = dict(experimento="extra_features_enso", variante=nome, n_features=len(feats),
             threshold=thr, f1=f1_score(y_te, yp), auc_roc=roc_auc_score(y_te, p),
             accuracy=accuracy_score(y_te, yp), precision=precision_score(y_te, yp),
             recall=recall_score(y_te, yp), train_size=len(train), test_size=len(test))
    resultados.append(r)
    importancias[nome] = sorted(zip(feats, m.feature_importances_), key=lambda x: -x[1])[:12]
    with mlflow.start_run(run_name=f"Exp_{nome}"):
        mlflow.log_params({"type": "experiment", "experimento": r["experimento"],
                           "variante": nome, "n_features": len(feats), "threshold": thr})
        mlflow.log_metrics({k: r[k] for k in ("f1", "auc_roc", "accuracy", "precision", "recall")})
    print(nome, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

res = pd.DataFrame(resultados)
res["created_at"] = datetime.now()
spark.createDataFrame(res).write.mode("append").option("mergeSchema", "true").saveAsTable(RESULTS_TABLE)

dbutils.notebook.exit(json.dumps({
    "resultados": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in resultados],
    "importancias_C": [(f, round(float(v), 4)) for f, v in importancias["C_base_era5_enso"]],
}))
