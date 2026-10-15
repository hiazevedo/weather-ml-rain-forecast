# Databricks notebook source
# Experimento: modelo regional treinado com chuva MEDIDA (INMET)
#
# O champion aprende a reproduzir o rótulo do Open-Meteo; contra a chuva
# medida ele fica abaixo da persistência do pluviômetro (notebook 11). Aqui o
# rótulo passa a ser o pluviômetro do INMET (chove na próxima hora?), com as
# mesmas 46 features do Open-Meteo no ponto de cada estação.
#
# Protocolo (simula Birigui, uma cidade SEM estação): para cada estação E
#   treino      — as OUTRAS estações, até 24 meses antes do fim do INMET
#   calibração  — as OUTRAS estações, de −24 a −12 meses (escolhe o limiar)
#   teste       — a estação E, últimos 12 meses
# Variantes avaliadas nas mesmas horas de teste:
#   champion             — modelo atual (rótulo Open-Meteo), no limiar dele
#   regional             — RandomForest com rótulo INMET
#   regional_mais_champ  — idem + probabilidade do champion como feature
#   persistencia         — "chove agora no pluviômetro → chove na próxima hora"
#                          (referência; não existe em Birigui, que não tem estação)

# COMMAND ----------

# MAGIC %run ./features

# COMMAND ----------

import json
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score, confusion_matrix)

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
UC_MODEL_NAME = "workspace.default.rain-forecast-birigui"
RAW = "/Volumes/weather_pipeline/bronze/station_points"
RESULTS_TABLE = "weather_pipeline.gold.ml_experiments"

client = mlflow.tracking.MlflowClient()
mv = client.get_model_version_by_alias(UC_MODEL_NAME, "champion")
champ = mlflow.sklearn.load_model(f"models:/{UC_MODEL_NAME}@champion")
thr_champ = float(mv.tags["threshold"])
FEATS = list(champ.feature_names_in_)

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
estacoes = sorted(inmet["station_id"].unique())

frames = []
for cod in estacoes:
    s = add_features_pandas(serie_om(cod))
    m = inmet[inmet.station_id == cod].set_index("observation_time")["precip_mm"].reindex(s.index)
    s["y"] = (m.shift(-1) > 0).astype(float).where(m.shift(-1).notna())      # chuva medida na próxima hora
    s["persist"] = (m > 0).astype(float).where(m.notna())
    s["station_id"] = cod
    frames.append(s.reset_index())
df = pd.concat(frames, ignore_index=True).dropna(subset=FEATS + ["y", "persist"])
df["p_champ"] = champ.predict_proba(df[FEATS])[:, 1]
fim = df["observation_time"].max()
c24, c12 = fim - pd.DateOffset(months=24), fim - pd.DateOffset(months=12)
print(f"{len(df):,} horas com features e chuva medida; fim {fim:%Y-%m-%d}; "
      f"chuva medida em {df['y'].mean():.1%} das horas")
print(df.groupby("station_id")["y"].agg(["size", "mean"]).round(3).to_string())

# COMMAND ----------

def limiar(y, p):
    return float(max(np.round(np.arange(0.05, 0.91, 0.05), 2),
                     key=lambda t: f1_score(y, (p >= t).astype(int), zero_division=0)))

VARIANTES = {"regional": FEATS, "regional_mais_champ": FEATS + ["p_champ"]}
preds = []                         # uma linha por hora de teste, com a previsão de cada variante
for est in estacoes:
    outras, alvo = df[df.station_id != est], df[df.station_id == est]
    tr = outras[outras.observation_time < c24]
    ca = outras[(outras.observation_time >= c24) & (outras.observation_time < c12)]
    te = alvo[alvo.observation_time >= c12].copy()
    if len(te) < 500:
        print(est, "pouco dado de teste:", len(te)); continue
    te["pred_champion"], te["prob_champion"] = (te["p_champ"] >= thr_champ).astype(int), te["p_champ"]
    te["pred_persistencia"] = te["persist"].astype(int)
    for nome, feats in VARIANTES.items():
        m = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                   class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        m.fit(tr[feats], tr["y"].astype(int))
        t = limiar(ca["y"].astype(int), m.predict_proba(ca[feats])[:, 1])
        te[f"prob_{nome}"] = m.predict_proba(te[feats])[:, 1]
        te[f"pred_{nome}"] = (te[f"prob_{nome}"] >= t).astype(int)
        te[f"thr_{nome}"] = t
    preds.append(te)
P = pd.concat(preds, ignore_index=True)

# COMMAND ----------

def metricas(y, yp, prob=None):
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return dict(f1=f1_score(y, yp, zero_division=0), precision=precision_score(y, yp, zero_division=0),
                recall=recall_score(y, yp), accuracy=accuracy_score(y, yp),
                auc_roc=roc_auc_score(y, prob) if prob is not None else None,
                vp=int(tp), fp=int(fp), fn=int(fn), vn=int(tn), positivos_previstos=float(np.mean(yp)))

linhas = []
for var in ("champion", "regional", "regional_mais_champ", "persistencia"):
    prob = P.get(f"prob_{var}")
    tot = metricas(P["y"].astype(int), P[f"pred_{var}"], prob)          # todas as estações juntas
    linhas.append(dict(variante=var, estacao="todas", n=len(P), **tot))
    for est, g in P.groupby("station_id"):
        linhas.append(dict(variante=var, estacao=est, n=len(g),
                           **metricas(g["y"].astype(int), g[f"pred_{var}"], g.get(f"prob_{var}"))))
R = pd.DataFrame(linhas)
print(R[R.estacao == "todas"].drop(columns="estacao").round(3).to_string(index=False))
print(R[R.estacao != "todas"].pivot(index="estacao", columns="variante", values="f1").round(3).to_string())

f = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
agora = datetime.now()
spark.createDataFrame(
    [("regional_inmet", f"{r['variante']}__{r['estacao']}", len(FEATS), None, f(r["f1"]), f(r["auc_roc"]),
      f(r["accuracy"]), f(r["precision"]), f(r["recall"]), None, int(r["n"]), agora) for _, r in R.iterrows()],
    "experimento string, variante string, n_features bigint, threshold double, f1 double, auc_roc double, "
    "accuracy double, `precision` double, recall double, train_size bigint, test_size bigint, created_at timestamp"
).write.mode("append").saveAsTable(RESULTS_TABLE)

dbutils.notebook.exit(json.dumps({
    "champion": f"v{mv.version} limiar {thr_champ}", "periodo_teste": [str(c12.date()), str(fim.date())],
    "chuva_medida_teste": round(float(P["y"].mean()), 4),
    "limiares_regional": P.groupby("station_id")[[c for c in P.columns if c.startswith("thr_")]].first().round(2).to_dict(),
    "todas": R[R.estacao == "todas"].drop(columns="estacao").round(4).to_dict("records"),
    "f1_por_estacao": R[R.estacao != "todas"].pivot(index="estacao", columns="variante", values="f1").round(3).to_dict()},
    default=str))
