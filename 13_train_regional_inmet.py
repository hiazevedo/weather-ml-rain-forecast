# Databricks notebook source
# Modelo REGIONAL — prevê chuva MEDIDA (pluviômetro do INMET) na próxima hora
#
# Roda em paralelo ao champion (que prevê o rótulo do Open-Meteo); não o substitui.
#   rótulo   : chuva > 0 no pluviômetro do INMET na hora seguinte
#   features : as 46 da variante B, do Open-Meteo no ponto de cada estação
#   estações : as do INMET num raio de 80 km de Birigui (silver.inmet_hourly)
#
# Etapas (experimento 12 mostrou o ganho: F1 contra chuva medida 0,328 → 0,427):
#   1. avaliação "deixando uma estação de fora" nos últimos 12 meses — é a métrica
#      honesta para uma cidade SEM estação, como Birigui;
#   2. limiar = mediana dos limiares calibrados em cada rodada (nas outras estações);
#   3. modelo final treinado com TODAS as estações e todo o período;
#   4. registro em workspace.default.rain-forecast-regional-inmet; o alias
#      "champion" passa para a nova versão se o F1 da avaliação não cair mais de
#      0,02 em relação ao champion regional atual.
#
# Retreinado mensalmente pelo job [weather-ml] Retreinamento Mensal.

# COMMAND ----------

# MAGIC %run ./features

# COMMAND ----------

import json
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.exceptions import MlflowException
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)

mlflow.autolog(disable=True)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/{}/weather-ml-rain-forecast".format(
    spark.sql("SELECT current_user()").collect()[0][0]))

MODEL_NAME = "workspace.default.rain-forecast-regional-inmet"
VERSIONS_TABLE = "weather_pipeline.gold.rain_regional_versions"
RAW = "/Volumes/weather_pipeline/bronze/station_points"   # Open-Meteo por estação (notebook 11)
HIPER = dict(n_estimators=300, max_depth=12, min_samples_leaf=20,
             class_weight="balanced_subsample", random_state=42, n_jobs=-1)

# COMMAND ----------

def serie_om(cod):
    fr = [pd.DataFrame(json.loads(dbutils.fs.head(f.path, 100_000_000)))
          for f in dbutils.fs.ls(RAW) if f.name.startswith(f"{cod}_")]
    d = pd.concat(fr, ignore_index=True).drop_duplicates("time")
    d["observation_time"] = pd.to_datetime(d.pop("time"))
    return d.rename(columns={"relative_humidity_2m": "relative_humidity", "windspeed_10m": "windspeed",
                             "total_column_integrated_water_vapour": "tcwv"})

inmet = spark.table("weather_pipeline.silver.inmet_hourly").select(
    "station_id", "observation_time", "precip_mm").toPandas()
inmet["observation_time"] = pd.to_datetime(inmet["observation_time"])
inmet = inmet.groupby(["station_id", "observation_time"], as_index=False)["precip_mm"].sum(min_count=1)
estacoes = sorted(set(inmet["station_id"]) & {f.name.split("_")[0] for f in dbutils.fs.ls(RAW)})

frames = []
for cod in estacoes:
    s = add_features_pandas(serie_om(cod))
    m = inmet[inmet.station_id == cod].set_index("observation_time")["precip_mm"].reindex(s.index)
    s["y"] = (m.shift(-1) > 0).astype(float).where(m.shift(-1).notna())
    s["station_id"] = cod
    frames.append(s.reset_index())
df = pd.concat(frames, ignore_index=True).dropna(subset=ML_FEATURES + ["y"])
df["y"] = df["y"].astype(int)
fim = df["observation_time"].max()
c24, c12 = fim - pd.DateOffset(months=24), fim - pd.DateOffset(months=12)
print(f"{len(df):,} horas de {len(estacoes)} estações {estacoes}; até {fim:%Y-%m-%d}; "
      f"chuva medida em {df['y'].mean():.1%} das horas")

# COMMAND ----------

# 1–2) Avaliação deixando uma estação de fora + limiar
def limiar(y, p):
    return float(max(np.round(np.arange(0.05, 0.91, 0.05), 2),
                     key=lambda t: f1_score(y, (p >= t).astype(int), zero_division=0)))

ys, ps, limiares = [], [], []
for est in estacoes:
    outras, alvo = df[df.station_id != est], df[df.station_id == est]
    tr = outras[outras.observation_time < c24]
    ca = outras[(outras.observation_time >= c24) & (outras.observation_time < c12)]
    te = alvo[alvo.observation_time >= c12]
    if len(te) < 500 or len(ca) < 500:
        print(est, "pouco dado — fora da avaliação"); continue
    m = RandomForestClassifier(**HIPER).fit(tr[ML_FEATURES], tr["y"])
    t = limiar(ca["y"], m.predict_proba(ca[ML_FEATURES])[:, 1])
    limiares.append(t)
    ys.append(te["y"].values); ps.append((m.predict_proba(te[ML_FEATURES])[:, 1], t))

threshold = float(np.median(limiares))
y_all = np.concatenate(ys)
p_all = np.concatenate([p for p, _ in ps])
yp_all = np.concatenate([(p >= t).astype(int) for p, t in ps])     # cada rodada no próprio limiar
met = dict(f1=f1_score(y_all, yp_all), precision=precision_score(y_all, yp_all),
           recall=recall_score(y_all, yp_all), accuracy=accuracy_score(y_all, yp_all),
           auc_roc=roc_auc_score(y_all, p_all), test_size=int(len(y_all)),
           test_rain_rate=float(y_all.mean()))
print(f"avaliação (estação fora, últimos 12 meses): "
      f"{ {k: round(v, 4) for k, v in met.items()} } | limiares {limiares} → {threshold}")

# COMMAND ----------

# 3–4) Modelo final com todas as estações, registro e alias
with mlflow.start_run(run_name="RF_Regional_INMET") as run:
    final = RandomForestClassifier(**HIPER).fit(df[ML_FEATURES], df["y"])
    mlflow.log_params({"model": "RandomForestClassifier", "type": "regional_inmet",
                       "rotulo": "chuva medida INMET (próxima hora)", "estacoes": ",".join(estacoes),
                       "train_size": len(df), "threshold": threshold, **{k: v for k, v in HIPER.items() if k != "n_jobs"}})
    mlflow.log_metrics({k: v for k, v in met.items() if k in ("f1", "precision", "recall", "accuracy", "auc_roc")})
    X_ex = df[ML_FEATURES].head(100)
    info = mlflow.sklearn.log_model(final, "model",
                                    signature=infer_signature(X_ex, final.predict(X_ex)))
    run_id = run.info.run_id

client = mlflow.tracking.MlflowClient()
mv = mlflow.register_model(info.model_uri, MODEL_NAME)
for k, v in {"run_id": run_id, "threshold": f"{threshold:.2f}", "f1_estacao_fora": f"{met['f1']:.4f}",
             "estacoes": ",".join(estacoes)}.items():
    client.set_model_version_tag(MODEL_NAME, mv.version, k, v)

try:
    atual = client.get_model_version_by_alias(MODEL_NAME, "champion")
    f1_atual = float(atual.tags.get("f1_estacao_fora", "0"))
except MlflowException as e:
    if e.error_code != "RESOURCE_DOES_NOT_EXIST":
        raise
    atual, f1_atual = None, None

# As avaliações de versões diferentes usam janelas de teste diferentes (o
# INMET avança mês a mês), então a regra tolera pequenas variações.
promote = atual is None or met["f1"] >= f1_atual - 0.02
if promote:
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
motivo = ("primeira versão" if atual is None else
          f"F1 {met['f1']:.4f} vs champion regional v{atual.version} {f1_atual:.4f} (tolerância 0,02)")

spark.createDataFrame([(int(mv.version), run_id, datetime.now(), bool(promote), threshold,
                        float(met["f1"]), float(met["precision"]), float(met["recall"]),
                        float(met["accuracy"]), float(met["auc_roc"]), int(len(df)),
                        met["test_size"], met["test_rain_rate"], ",".join(estacoes))],
    "version int, run_id string, created_at timestamp, promoted boolean, threshold double, "
    "f1 double, `precision` double, recall double, accuracy double, auc_roc double, "
    "train_size int, test_size int, test_rain_rate double, estacoes string"
).write.mode("append").saveAsTable(VERSIONS_TABLE)

print(f"{MODEL_NAME} v{mv.version} registrada | limiar {threshold:.2f} | "
      f"champion regional {'PROMOVIDO' if promote else 'mantido'} — {motivo}")
dbutils.notebook.exit(json.dumps({"version": int(mv.version), "promoted": promote, "threshold": threshold,
                                  "metricas": {k: round(v, 4) for k, v in met.items()}, "motivo": motivo}))
