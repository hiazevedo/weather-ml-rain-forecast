# Databricks notebook source
# Experimento: corrigir a previsão (MOS) com previsões arquivadas
#
# Todas as abordagens tentam prever "chove na hora t?" (chuva > 0 no dado
# observado da silver — ERA5/Archive) nas MESMAS horas dos últimos 12 meses:
#   api_d0 / api_d1  — a previsão da Open-Meteo diz > 0 mm (rodada mais
#                      recente / de 1 dia antes)
#   champion_prev    — o modelo atual alimentado com a previsão (como ocorre
#                      em produção para as horas futuras): features da hora
#                      t−1 calculadas sobre a série prevista → chuva em t
#   mos_d0 / mos_d1  — RandomForest treinado com as previsões (d0 ou d1) da
#                      própria hora t e vizinhas, + hora/mês + ENSO
#
# Dados: silver.forecast_previous_runs (Previous Runs API, desde 2024).
# Nada aqui altera o champion.

# COMMAND ----------

import json
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                             precision_score, recall_score)

mlflow.autolog(disable=True)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
UC_MODEL_NAME = "workspace.default.rain-forecast-birigui"
RESULTS_TABLE = "weather_pipeline.gold.ml_experiments"
VARS = ["precipitation", "temperature_2m", "apparent_temperature", "relative_humidity_2m",
        "dew_point_2m", "pressure_msl", "cloud_cover", "wind_speed_10m", "wind_direction_10m"]
CHAMP_FEATURES = [
    "temperature_2m", "apparent_temperature", "precipitation", "relative_humidity", "windspeed",
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

# COMMAND ----------

# Série horária contínua: previsões arquivadas + chuva observada + ONI do mês
prev = spark.table("weather_pipeline.silver.forecast_previous_runs").toPandas()
obs = spark.sql("""
    SELECT observation_time, precipitation AS chuva_obs
    FROM weather_pipeline.silver.weather_clean WHERE is_forecast = false
""").toPandas()
oni = spark.sql("""
    SELECT available_month AS mes, oni FROM weather_pipeline.silver.enso_oni
""").toPandas()

prev["observation_time"] = pd.to_datetime(prev["observation_time"])
obs["observation_time"] = pd.to_datetime(obs["observation_time"])
idx = pd.date_range(prev["observation_time"].min(), prev["observation_time"].max(), freq="h")
df = (prev.set_index("observation_time").reindex(idx)
        .join(obs.set_index("observation_time"), how="left"))
df.index.name = "observation_time"
df = df.reset_index()
df["mes"] = df["observation_time"].dt.to_period("M").dt.to_timestamp()
oni["mes"] = pd.to_datetime(oni["mes"])
meses = pd.DataFrame({"mes": pd.date_range(oni["mes"].min(), df["mes"].max(), freq="MS")})
oni_mes = meses.merge(oni, on="mes", how="left").ffill()      # último índice publicado
df = df.merge(oni_mes, on="mes", how="left")
df["y"] = (df["chuva_obs"] > 0).astype("float").where(df["chuva_obs"].notna())

# COMMAND ----------

def features_mos(d, lead):
    """Previsões (antecedência `lead`) da hora t e vizinhas, para prever a hora t."""
    X = pd.DataFrame(index=d.index)
    for v in VARS:
        X[f"{v}"] = d[f"{v}_d{lead}"]
    p = d[f"precipitation_d{lead}"]
    for k in (1, 2, 3):                       # erro de fase: a chuva prevista pode adiantar/atrasar
        X[f"precip_lag_{k}"]  = p.shift(k)
        X[f"precip_lead_{k}"] = p.shift(-k)
    X["precip_win_7h"] = p.rolling(7, center=True, min_periods=7).sum()
    X["pressure_tend_3h"] = d[f"pressure_msl_d{lead}"].diff(3)
    X["dewpoint_depression"] = d[f"temperature_2m_d{lead}"] - d[f"dew_point_2m_d{lead}"]
    X["wind_dir_sin"] = np.sin(np.radians(d[f"wind_direction_10m_d{lead}"]))
    X["wind_dir_cos"] = np.cos(np.radians(d[f"wind_direction_10m_d{lead}"]))
    h, m = d["observation_time"].dt.hour, d["observation_time"].dt.month
    X["hour_sin"], X["hour_cos"] = np.sin(h * 2 * np.pi / 24), np.cos(h * 2 * np.pi / 24)
    X["month_sin"], X["month_cos"] = np.sin(m * 2 * np.pi / 12), np.cos(m * 2 * np.pi / 12)
    X["oni"] = d["oni"]
    return X

def features_champion(d, lead=0):
    """As 31 features do champion calculadas sobre a série PREVISTA (como em produção)."""
    s = pd.DataFrame({
        "temperature_2m": d[f"temperature_2m_d{lead}"], "apparent_temperature": d[f"apparent_temperature_d{lead}"],
        "precipitation": d[f"precipitation_d{lead}"], "relative_humidity": d[f"relative_humidity_2m_d{lead}"],
        "windspeed": d[f"wind_speed_10m_d{lead}"]})
    t, p, u = s["temperature_2m"], s["precipitation"], s["relative_humidity"]
    for k in (1, 3, 24): s[f"temp_lag_{k}h"] = t.shift(k); s[f"precip_lag_{k}h"] = p.shift(k)
    for k in (1, 3): s[f"humidity_lag_{k}h"] = u.shift(k)
    s["temp_delta_1h"], s["humidity_delta_1h"], s["precip_delta_1h"] = t.diff(), u.diff(), p.diff()
    s["temp_rolling_6h"], s["humidity_rolling_6h"] = t.rolling(6).mean(), u.rolling(6).mean()
    for k in (3, 6, 24): s[f"precip_rolling_{k}h"] = p.rolling(k).sum()
    h, m = d["observation_time"].dt.hour, d["observation_time"].dt.month
    s["hour"], s["month"] = h, m
    s["weekday"] = (d["observation_time"].dt.dayofweek + 1) % 7 + 1       # mesma convenção do Spark
    s["hour_sin"], s["hour_cos"] = np.sin(h * 2 * np.pi / 24), np.cos(h * 2 * np.pi / 24)
    s["month_sin"], s["month_cos"] = np.sin(m * 2 * np.pi / 12), np.cos(m * 2 * np.pi / 12)
    s["is_rainy_season"] = m.isin([10, 11, 12, 1, 2, 3]).astype(int)
    s["is_peak_rain_month"] = m.isin([1, 2, 12]).astype(int)
    s["is_afternoon"] = h.between(12, 18).astype(int)
    # o champion usa as features da hora t−1 para prever t
    return s[CHAMP_FEATURES].shift(1)

# COMMAND ----------

# d0 (rodada mais recente) é praticamente igual ao "observado" recente da silver
# (ambos vêm das primeiras horas das rodadas do IFS) — serve só de checagem.
# As comparações úteis são com antecedência de 1 e 2 dias (d1, d2).
LEADS = (0, 1, 2)
XM = {l: features_mos(df, l) for l in LEADS}
XC = {l: features_champion(df, l) for l in LEADS}
ok = df["y"].notna()
for l in LEADS:
    ok &= XM[l].notna().all(axis=1) & XC[l].notna().all(axis=1)
d = df[ok].reset_index(drop=True)
XM = {l: x[ok].reset_index(drop=True) for l, x in XM.items()}
XC = {l: x[ok].reset_index(drop=True) for l, x in XC.items()}
y = d["y"].astype(int).values

t = d["observation_time"]
fim = t.max()
test_ini = fim - pd.DateOffset(months=12)
cal_ini = test_ini - pd.DateOffset(months=4)
tr, ca, te = (t < cal_ini).values, ((t >= cal_ini) & (t < test_ini)).values, (t >= test_ini).values
print(f"horas: treino {tr.sum():,} | calibração {ca.sum():,} | teste {te.sum():,} "
      f"({t.min():%Y-%m-%d} → {fim:%Y-%m-%d}); chuva no teste: {y[te].mean():.1%}")

def metricas(nome, yp, score=None, thr=None):
    r = dict(experimento="mos_previous_runs", variante=nome, threshold=thr,
             f1=f1_score(y[te], yp), accuracy=accuracy_score(y[te], yp),
             precision=precision_score(y[te], yp, zero_division=0),
             recall=recall_score(y[te], yp),
             auc_roc=roc_auc_score(y[te], score) if score is not None else None,
             train_size=int(tr.sum()), test_size=int(te.sum()))
    print(nome, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    return r

resultados = []
# API pura
for lead in LEADS:
    pr = d[f"precipitation_d{lead}"].values[te]
    resultados.append(metricas(f"api_d{lead}", (pr > 0).astype(int), score=pr))

# Champion alimentado com a previsão
client = mlflow.tracking.MlflowClient()
mv = client.get_model_version_by_alias(UC_MODEL_NAME, "champion")
champ = mlflow.sklearn.load_model(f"models:/{UC_MODEL_NAME}@champion")
thr_c = float(mv.tags["threshold"])
for lead in LEADS:
    pc = champ.predict_proba(XC[lead][te])[:, 1]
    resultados.append(metricas(f"champion_v{mv.version}_d{lead}", (pc >= thr_c).astype(int), score=pc, thr=thr_c))

# MOS
importancias = {}
for lead in (1, 2):
    X = XM[lead]
    m = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                               class_weight="balanced", random_state=42, n_jobs=-1)
    m.fit(X[tr], y[tr])
    pcal = m.predict_proba(X[ca])[:, 1]
    thr = float(max(np.round(np.arange(0.30, 0.91, 0.05), 2),
                    key=lambda th: f1_score(y[ca], (pcal >= th).astype(int))))
    p = m.predict_proba(X[te])[:, 1]
    resultados.append(metricas(f"mos_d{lead}", (p >= thr).astype(int), score=p, thr=thr))
    importancias[f"mos_d{lead}"] = sorted(zip(X.columns, m.feature_importances_), key=lambda x: -x[1])[:10]

for r in resultados:
    with mlflow.start_run(run_name=f"Exp_MOS_{r['variante']}"):
        mlflow.log_params({"type": "experiment", "experimento": r["experimento"],
                           "variante": r["variante"], "threshold": r["threshold"]})
        mlflow.log_metrics({k: r[k] for k in ("f1", "accuracy", "precision", "recall", "auc_roc")
                            if r[k] is not None})

f = lambda v: None if v is None else float(v)
agora = datetime.now()
linhas = [(r["experimento"], r["variante"], None, f(r["threshold"]), f(r["f1"]), f(r["auc_roc"]),
           f(r["accuracy"]), f(r["precision"]), f(r["recall"]), r["train_size"], r["test_size"], agora)
          for r in resultados]
spark.createDataFrame(linhas,
    "experimento string, variante string, n_features bigint, threshold double, f1 double, "
    "auc_roc double, accuracy double, `precision` double, recall double, train_size bigint, "
    "test_size bigint, created_at timestamp"
).write.mode("append").saveAsTable(RESULTS_TABLE)

dbutils.notebook.exit(json.dumps({
    "periodo_teste": [str(test_ini.date()), str(fim.date())], "chuva_teste": round(float(y[te].mean()), 4),
    "resultados": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in resultados],
    "importancias": {k: [(f, round(float(v), 4)) for f, v in vs] for k, vs in importancias.items()},
}))
