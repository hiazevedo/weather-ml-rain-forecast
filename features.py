# Databricks notebook source
# Features do modelo de chuva — definição única, usada via `%run ./features`
# por 01_feature_engineering, 07_update_features, 05_batch_inference,
# 06_retrain_model e 03/04.
#
# Variante B (experimento 08): as 31 features originais + 15 derivadas das
# variáveis extras do ERA5/Open-Meteo (orvalho, pressão, nuvens por camada,
# direção do vento, vapor d'água na coluna, camada limite).

# COMMAND ----------

import math
from pyspark.sql import functions as F

ML_FEATURES_BASE = [
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
    # Temporais cíclicas e brutas
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "hour", "month", "weekday",
    # Flags
    "is_rainy_season", "is_peak_rain_month", "is_afternoon",
]

ML_FEATURES_EXTRA = [
    "dew_point_2m", "dewpoint_depression",
    "pressure_msl", "pressure_tend_3h", "pressure_tend_24h",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "cloud_low_delta_1h",
    "wind_dir_sin", "wind_dir_cos",
    "tcwv", "tcwv_delta_3h",
    "boundary_layer_height",
]

ML_FEATURES = ML_FEATURES_BASE + ML_FEATURES_EXTRA

# Colunas da silver necessárias para calcular as features
SILVER_COLS = [
    "observation_time", "temperature_2m", "apparent_temperature",
    "precipitation", "relative_humidity", "windspeed",
    "dew_point_2m", "pressure_msl", "cloud_cover", "cloud_cover_low",
    "cloud_cover_mid", "cloud_cover_high", "wind_direction_10m", "tcwv",
    "boundary_layer_height",
    "year", "month", "day", "hour", "weekday",
]

# COMMAND ----------

def add_features(df, w):
    """Acrescenta as features a uma série horária ordenada pela janela `w`.

    Lags/rolling são por posição de linha: a série precisa ser contínua e sem
    horas duplicadas (a silver já é deduplicada por observation_time).
    Requer as colunas de SILVER_COLS (exceto year/day, que não são features).
    """
    PI = math.pi
    lag = lambda c, k: F.lag(c, k).over(w)
    return (
        df
        # Lag
        .withColumn("temp_lag_1h",       lag("temperature_2m", 1))
        .withColumn("temp_lag_3h",       lag("temperature_2m", 3))
        .withColumn("temp_lag_24h",      lag("temperature_2m", 24))
        .withColumn("precip_lag_1h",     lag("precipitation", 1))
        .withColumn("precip_lag_3h",     lag("precipitation", 3))
        .withColumn("precip_lag_24h",    lag("precipitation", 24))
        .withColumn("humidity_lag_1h",   lag("relative_humidity", 1))
        .withColumn("humidity_lag_3h",   lag("relative_humidity", 3))
        # Delta
        .withColumn("temp_delta_1h",     F.col("temperature_2m") - lag("temperature_2m", 1))
        .withColumn("humidity_delta_1h", F.col("relative_humidity") - lag("relative_humidity", 1))
        .withColumn("precip_delta_1h",   F.col("precipitation") - lag("precipitation", 1))
        # Rolling
        .withColumn("temp_rolling_6h",     F.round(F.avg("temperature_2m").over(w.rowsBetween(-5, 0)), 3))
        .withColumn("humidity_rolling_6h", F.round(F.avg("relative_humidity").over(w.rowsBetween(-5, 0)), 3))
        .withColumn("precip_rolling_3h",   F.round(F.sum("precipitation").over(w.rowsBetween(-2, 0)), 3))
        .withColumn("precip_rolling_6h",   F.round(F.sum("precipitation").over(w.rowsBetween(-5, 0)), 3))
        .withColumn("precip_rolling_24h",  F.round(F.sum("precipitation").over(w.rowsBetween(-23, 0)), 3))
        # Encoding cíclico
        .withColumn("hour_sin",  F.round(F.sin(F.col("hour") * (2 * PI / 24)), 6))
        .withColumn("hour_cos",  F.round(F.cos(F.col("hour") * (2 * PI / 24)), 6))
        .withColumn("month_sin", F.round(F.sin(F.col("month") * (2 * PI / 12)), 6))
        .withColumn("month_cos", F.round(F.cos(F.col("month") * (2 * PI / 12)), 6))
        # Flags sazonais
        .withColumn("is_rainy_season",    F.when(F.col("month").isin(10, 11, 12, 1, 2, 3), 1).otherwise(0))
        .withColumn("is_peak_rain_month", F.when(F.col("month").isin(1, 2, 12), 1).otherwise(0))
        .withColumn("is_afternoon",       F.when(F.col("hour").between(12, 18), 1).otherwise(0))
        # Extras (variante B)
        .withColumn("dewpoint_depression", F.col("temperature_2m") - F.col("dew_point_2m"))
        .withColumn("pressure_tend_3h",    F.col("pressure_msl") - lag("pressure_msl", 3))
        .withColumn("pressure_tend_24h",   F.col("pressure_msl") - lag("pressure_msl", 24))
        .withColumn("cloud_low_delta_1h",  F.col("cloud_cover_low") - lag("cloud_cover_low", 1))
        .withColumn("tcwv_delta_3h",       F.col("tcwv") - lag("tcwv", 3))
        .withColumn("wind_dir_sin",        F.sin(F.radians("wind_direction_10m")))
        .withColumn("wind_dir_cos",        F.cos(F.radians("wind_direction_10m")))
    )

def add_features_pandas(h):
    """Mesmas features de add_features(), em pandas, para séries de outros pontos
    (ex.: Open-Meteo nas coordenadas das estações do INMET).

    `h`: DataFrame com observation_time (hora local) e as colunas brutas
    (temperature_2m, apparent_temperature, precipitation, relative_humidity,
    windspeed, dew_point_2m, pressure_msl, cloud_cover*, wind_direction_10m,
    tcwv, boundary_layer_height). Devolve a série horária contínua (horas
    faltantes viram NaN), indexada por observation_time.
    """
    import numpy as np
    import pandas as pd
    idx = pd.date_range(h["observation_time"].min(), h["observation_time"].max(), freq="h")
    s = h.drop_duplicates("observation_time").set_index("observation_time").reindex(idx)
    s.index.name = "observation_time"
    t, p, u = s["temperature_2m"], s["precipitation"], s["relative_humidity"]
    for k in (1, 3, 24):
        s[f"temp_lag_{k}h"], s[f"precip_lag_{k}h"] = t.shift(k), p.shift(k)
    for k in (1, 3):
        s[f"humidity_lag_{k}h"] = u.shift(k)
    s["temp_delta_1h"], s["humidity_delta_1h"], s["precip_delta_1h"] = t.diff(), u.diff(), p.diff()
    s["temp_rolling_6h"] = t.rolling(6).mean().round(3)
    s["humidity_rolling_6h"] = u.rolling(6).mean().round(3)
    for k in (3, 6, 24):
        s[f"precip_rolling_{k}h"] = p.rolling(k).sum().round(3)
    hr, mo = s.index.hour, s.index.month
    s["hour"], s["month"], s["weekday"] = hr, mo, (s.index.dayofweek + 1) % 7 + 1   # convenção do Spark
    s["hour_sin"], s["hour_cos"] = np.round(np.sin(hr * 2 * np.pi / 24), 6), np.round(np.cos(hr * 2 * np.pi / 24), 6)
    s["month_sin"], s["month_cos"] = np.round(np.sin(mo * 2 * np.pi / 12), 6), np.round(np.cos(mo * 2 * np.pi / 12), 6)
    s["is_rainy_season"] = np.isin(mo, [10, 11, 12, 1, 2, 3]).astype(int)
    s["is_peak_rain_month"] = np.isin(mo, [1, 2, 12]).astype(int)
    s["is_afternoon"] = ((hr >= 12) & (hr <= 18)).astype(int)
    s["dewpoint_depression"] = t - s["dew_point_2m"]
    s["pressure_tend_3h"], s["pressure_tend_24h"] = s["pressure_msl"].diff(3), s["pressure_msl"].diff(24)
    s["cloud_low_delta_1h"] = s["cloud_cover_low"].diff()
    s["tcwv_delta_3h"] = s["tcwv"].diff(3)
    s["wind_dir_sin"] = np.sin(np.radians(s["wind_direction_10m"]))
    s["wind_dir_cos"] = np.cos(np.radians(s["wind_direction_10m"]))
    return s

print(f"features carregadas — {len(ML_FEATURES)} ({len(ML_FEATURES_BASE)} base + {len(ML_FEATURES_EXTRA)} extras)")
