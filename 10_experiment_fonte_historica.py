# Databricks notebook source
# Experimento: qual fonte histórica usar para treinar?
#
# O modo padrão do Archive ("best_match") é ERA5 puro até 2017 e mistura o
# modelo IFS (ECMWF) a partir de 2018; o "observado" recente que a produção
# usa é IFS. Comparamos dois modelos (features da variante B):
#   treino_bestmatch — série atual (mistura)
#   treino_era5      — ERA5 puro de 1940 até hoje (fonte única)
# Cada um calibra o limiar na própria fonte e é testado nos últimos 12 meses:
#   teste_bestmatch  — o que a produção vê (critério principal)
#   teste_era5       — consistência com o ERA5
#
# Os arquivos ERA5 ficam em bronze.era5_only no mesmo formato dos
# historical_<ano>.json de produção, para reaproveitamento se o ERA5 vencer.

# COMMAND ----------

import json, time
from datetime import date, timedelta, datetime, timezone
import numpy as np
import pandas as pd
import requests
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                             precision_score, recall_score)

mlflow.autolog(disable=True)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/{}/weather-ml-rain-forecast".format(
    spark.sql("SELECT current_user()").collect()[0][0]))

LAT, LON, TZ, CIDADE = -21.2878, -50.3409, "America/Sao_Paulo", "Birigui-SP"
BASE_VARS = ["temperature_2m", "apparent_temperature", "precipitation", "rain",
             "relative_humidity_2m", "windspeed_10m"]
EXTRA_VARS = ["dew_point_2m", "pressure_msl", "cloud_cover", "cloud_cover_low",
              "cloud_cover_mid", "cloud_cover_high", "wind_direction_10m",
              "total_column_integrated_water_vapour", "boundary_layer_height"]
spark.sql("CREATE VOLUME IF NOT EXISTS weather_pipeline.bronze.era5_only")
RAW = "/Volumes/weather_pipeline/bronze/era5_only"
RESULTS_TABLE = "weather_pipeline.gold.ml_experiments"
fim = date.today() - timedelta(days=5)          # atraso do Archive

# COMMAND ----------

# 1) ERA5 puro (models=era5), todas as variáveis, um arquivo por ano
def fetch(start, end, tentativas=8):
    for i in range(tentativas):
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": LAT, "longitude": LON, "start_date": start, "end_date": end,
            "hourly": ",".join(BASE_VARS + EXTRA_VARS), "timezone": TZ, "models": "era5"},
            timeout=180)
        if r.status_code == 429:
            time.sleep(60 * (i + 1)); continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Archive recusou {start}..{end}")

existentes = {f.name for f in dbutils.fs.ls(RAW)}
for ano in range(1940, fim.year + 1):
    nome = f"historical_{ano}.json"
    if nome in existentes and ano < fim.year:
        continue
    data = fetch(f"{ano}-01-01", min(date(ano, 12, 31), fim).isoformat())
    payload = {"collected_at": datetime.now(timezone.utc).isoformat(), "cidade": CIDADE,
               "latitude": LAT, "longitude": LON, "data": data}
    dbutils.fs.put(f"{RAW}/{nome}", json.dumps(payload), overwrite=True)
    time.sleep(6)

era5 = pd.concat([pd.DataFrame(json.loads(dbutils.fs.head(f.path, 100_000_000))["data"]["hourly"])
                  for f in dbutils.fs.ls(RAW)], ignore_index=True).drop_duplicates("time")
era5["observation_time"] = pd.to_datetime(era5.pop("time"))
era5 = era5.rename(columns={"relative_humidity_2m": "relative_humidity", "windspeed_10m": "windspeed"})
print(f"ERA5: {len(era5):,} horas")

# COMMAND ----------

# 2) Série atual (best_match): silver (até o fim do Archive) + variáveis extras
bm = spark.sql(f"""
    SELECT c.observation_time, c.temperature_2m, c.apparent_temperature, c.precipitation,
           c.relative_humidity, c.windspeed, x.* EXCEPT (observation_time)
    FROM weather_pipeline.silver.weather_clean c
    JOIN weather_pipeline.silver.era5_extra_hourly x USING (observation_time)
    WHERE NOT c.is_forecast AND c.observation_time < DATE'{fim + timedelta(days=1)}'
""").toPandas()
bm["observation_time"] = pd.to_datetime(bm["observation_time"])
print(f"best_match: {len(bm):,} horas")

# COMMAND ----------

FEATURES = [
    # base (31)
    "temperature_2m", "apparent_temperature", "precipitation", "relative_humidity", "windspeed",
    "temp_lag_1h", "temp_lag_3h", "temp_lag_24h", "precip_lag_1h", "precip_lag_3h", "precip_lag_24h",
    "humidity_lag_1h", "humidity_lag_3h", "temp_delta_1h", "humidity_delta_1h", "precip_delta_1h",
    "temp_rolling_6h", "humidity_rolling_6h", "precip_rolling_3h", "precip_rolling_6h", "precip_rolling_24h",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "hour", "month", "weekday",
    "is_rainy_season", "is_peak_rain_month", "is_afternoon",
    # extras ERA5 (15)
    "dew_point_2m", "dewpoint_depression", "pressure_msl", "pressure_tend_3h", "pressure_tend_24h",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "cloud_low_delta_1h",
    "wind_dir_sin", "wind_dir_cos", "tcwv", "tcwv_delta_3h", "boundary_layer_height",
]

def build(h):
    """Features da variante B + alvo (chuva na hora seguinte), numa série horária contínua."""
    idx = pd.date_range(h["observation_time"].min(), h["observation_time"].max(), freq="h")
    s = h.set_index("observation_time").reindex(idx)
    t, p, u = s["temperature_2m"], s["precipitation"], s["relative_humidity"]
    for k in (1, 3, 24): s[f"temp_lag_{k}h"] = t.shift(k); s[f"precip_lag_{k}h"] = p.shift(k)
    for k in (1, 3): s[f"humidity_lag_{k}h"] = u.shift(k)
    s["temp_delta_1h"], s["humidity_delta_1h"], s["precip_delta_1h"] = t.diff(), u.diff(), p.diff()
    s["temp_rolling_6h"], s["humidity_rolling_6h"] = t.rolling(6).mean(), u.rolling(6).mean()
    for k in (3, 6, 24): s[f"precip_rolling_{k}h"] = p.rolling(k).sum()
    hr, mo = s.index.hour, s.index.month
    s["hour"], s["month"], s["weekday"] = hr, mo, (s.index.dayofweek + 1) % 7 + 1
    s["hour_sin"], s["hour_cos"] = np.sin(hr * 2 * np.pi / 24), np.cos(hr * 2 * np.pi / 24)
    s["month_sin"], s["month_cos"] = np.sin(mo * 2 * np.pi / 12), np.cos(mo * 2 * np.pi / 12)
    s["is_rainy_season"] = np.isin(mo, [10, 11, 12, 1, 2, 3]).astype(int)
    s["is_peak_rain_month"] = np.isin(mo, [1, 2, 12]).astype(int)
    s["is_afternoon"] = ((hr >= 12) & (hr <= 18)).astype(int)
    s["dewpoint_depression"] = t - s["dew_point_2m"]
    s["pressure_tend_3h"], s["pressure_tend_24h"] = s["pressure_msl"].diff(3), s["pressure_msl"].diff(24)
    s["cloud_low_delta_1h"] = s["cloud_cover_low"].diff()
    s["tcwv"] = s["total_column_integrated_water_vapour"]
    s["tcwv_delta_3h"] = s["tcwv"].diff(3)
    s["wind_dir_sin"] = np.sin(np.radians(s["wind_direction_10m"]))
    s["wind_dir_cos"] = np.cos(np.radians(s["wind_direction_10m"]))
    nxt = p.shift(-1)
    s["y"] = (nxt > 0).astype(float).where(nxt.notna())
    return s[FEATURES + ["y"]].dropna()

F_bm, F_e5 = build(bm), build(era5)
comum = F_bm.index.intersection(F_e5.index)          # mesmas horas nas duas fontes
F_bm, F_e5 = F_bm.loc[comum], F_e5.loc[comum]
c24 = pd.Timestamp(date.today()) - pd.DateOffset(months=24)
c12 = pd.Timestamp(date.today()) - pd.DateOffset(months=12)
tr, ca, te = comum < c24, (comum >= c24) & (comum < c12), comum >= c12
print(f"horas: treino {tr.sum():,} | calibração {ca.sum():,} | teste {te.sum():,}")
print(f"chuva na hora seguinte (teste): best_match {F_bm['y'][te].mean():.1%} | era5 {F_e5['y'][te].mean():.1%}")
print(f"rótulo igual nas duas fontes (teste): {(F_bm['y'][te] == F_e5['y'][te]).mean():.1%}")

# COMMAND ----------

resultados = []
fontes = {"bestmatch": F_bm, "era5": F_e5}
for treino, Ftr in fontes.items():
    m = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced",
                               random_state=42, n_jobs=-1)
    m.fit(Ftr.loc[tr, FEATURES], Ftr.loc[tr, "y"].astype(int))
    pcal = m.predict_proba(Ftr.loc[ca, FEATURES])[:, 1]
    ycal = Ftr.loc[ca, "y"].astype(int)
    thr = float(max(np.round(np.arange(0.30, 0.91, 0.05), 2),
                    key=lambda th: f1_score(ycal, (pcal >= th).astype(int))))
    for teste, Fte in fontes.items():
        yt = Fte.loc[te, "y"].astype(int).values
        p = m.predict_proba(Fte.loc[te, FEATURES])[:, 1]
        yp = (p >= thr).astype(int)
        r = dict(experimento="fonte_historica", variante=f"treino_{treino}__teste_{teste}",
                 n_features=len(FEATURES), threshold=thr, f1=f1_score(yt, yp),
                 auc_roc=roc_auc_score(yt, p), accuracy=accuracy_score(yt, yp),
                 precision=precision_score(yt, yp), recall=recall_score(yt, yp),
                 train_size=int(tr.sum()), test_size=int(te.sum()))
        resultados.append(r)
        print(r["variante"], {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
        with mlflow.start_run(run_name=f"Exp_Fonte_{r['variante']}"):
            mlflow.log_params({"type": "experiment", "experimento": r["experimento"],
                               "variante": r["variante"], "threshold": thr})
            mlflow.log_metrics({k: r[k] for k in ("f1", "auc_roc", "accuracy", "precision", "recall")})

agora = datetime.now()
spark.createDataFrame(
    [(r["experimento"], r["variante"], r["n_features"], r["threshold"], float(r["f1"]), float(r["auc_roc"]),
      float(r["accuracy"]), float(r["precision"]), float(r["recall"]), r["train_size"], r["test_size"], agora)
     for r in resultados],
    "experimento string, variante string, n_features bigint, threshold double, f1 double, "
    "auc_roc double, accuracy double, `precision` double, recall double, train_size bigint, "
    "test_size bigint, created_at timestamp").write.mode("append").saveAsTable(RESULTS_TABLE)

dbutils.notebook.exit(json.dumps({
    "chuva_teste": {k: round(float(v["y"][te].mean()), 4) for k, v in fontes.items()},
    "rotulo_igual_teste": round(float((F_bm["y"][te] == F_e5["y"][te]).mean()), 4),
    "resultados": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in resultados]}))
