# Databricks notebook source
# MOS multimodelo — treino (mensal) do modelo que prevê chuva MEDIDA na hora a
# partir das previsões de 6 modelos (experimento 14: F1 contra chuva medida
# 0,256 da API → 0,307, AUC ~0,67 → ~0,84, com cada estação fora do treino).
#
#   rótulo   : chuva > 0 no pluviômetro das estações do INMET num raio de 80 km
#   features : ./mos (6 modelos: chuva na hora e vizinhas, nuvens, umidade, CAPE,
#              consenso), da rodada de 1 dia antes (Previous Runs API)
#   avaliação: cada estação fora do treino, nos últimos 12 meses com rótulo, ao
#              lado da previsão crua da API (best_match > 0) nas mesmas horas
#   registro : workspace.default.rain-forecast-mos-multimodelo (MLflow, Unity
#              Catalog); o alias "champion" passa para a nova versão se ela bater a
#              API e o F1 não cair mais de 0,02 em relação ao champion atual.
#              Métricas por versão também em gold.rain_mos_versions (painel).
#
# MLflow ≥ 3: com o 2.11 (ambiente padrão de execuções avulsas no serverless) o
# upload dos artefatos ao Unity Catalog é negado (S3 AccessDenied).
#
# Produção: 17_inference_mos_multimodelo (Forecast API, a cada 3 h).

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.1"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./mos

# COMMAND ----------

import json, time
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import requests
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.exceptions import MlflowException
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             roc_auc_score, brier_score_loss)

mlflow.autolog(disable=True)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")   # sem isso o MLflow lê uma config bloqueada no serverless
mlflow.set_experiment("/Users/{}/weather-ml-rain-forecast".format(
    spark.sql("SELECT current_user()").collect()[0][0]))

MODEL_NAME = "workspace.default.rain-forecast-mos-multimodelo"
VERSIONS_TABLE = "weather_pipeline.gold.rain_mos_versions"
RAW = "/Volumes/weather_pipeline/bronze/previous_runs_multi"   # cache mensal (experimento 14)
LEAD = 1
INICIO = date(2024, 3, 1)
HIPER = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
LIMIARES = np.round(np.arange(0.05, 0.96, 0.05), 2)

est = spark.sql("""SELECT station_id, first(lat) AS lat, first(lon) AS lon
                   FROM weather_pipeline.silver.inmet_hourly GROUP BY station_id""").toPandas()

# COMMAND ----------

# Atualiza o cache da Previous Runs nas estações. O IP do Databricks às vezes é
# recusado (429) por horas: nesse caso o treino segue com o cache (o rótulo do
# INMET também chega com atraso) e o aviso fica no log.
hourly = [f"{v}_previous_day{l}" for v in MOS_VARS for l in (1, 2)]
fim = date.today() - timedelta(days=5)
existentes = {f.name: f.modificationTime / 1000 for f in dbutils.fs.ls(RAW)}
recusado = False
for r in est.itertuples():
    mes = INICIO
    while mes <= fim and not recusado:
        prox = (mes.replace(day=28) + timedelta(days=4)).replace(day=1)
        nome = f"{r.station_id}_{mes:%Y_%m}.json"
        recente = time.time() - existentes.get(nome, 0) < 24 * 3600
        if not (nome in existentes and (prox <= fim or recente)):
            resp = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast", params={
                "latitude": r.lat, "longitude": r.lon, "start_date": mes.isoformat(),
                "end_date": min(prox - timedelta(days=1), fim).isoformat(), "hourly": ",".join(hourly),
                "models": ",".join(MOS_MODELOS), "timezone": "America/Sao_Paulo"}, timeout=180)
            if resp.status_code == 429:
                print(f"⚠️ Previous Runs recusou ({nome}); treinando com o cache"); recusado = True; break
            resp.raise_for_status()
            dbutils.fs.put(f"{RAW}/{nome}", json.dumps(resp.json()["hourly"]), overwrite=True)
            time.sleep(3)
        mes = prox

def serie(ponto):
    fr = [pd.DataFrame(json.loads(dbutils.fs.head(f.path, 100_000_000)))
          for f in dbutils.fs.ls(RAW) if f.name.startswith(f"{ponto}_")]
    d = pd.concat(fr, ignore_index=True).drop_duplicates("time")
    d["observation_time"] = pd.to_datetime(d.pop("time"))
    return d.set_index("observation_time").sort_index()

inmet = spark.sql("""
    SELECT station_id, observation_time, sum(precip_mm) AS chuva FROM weather_pipeline.silver.inmet_hourly
    WHERE observation_time >= '2024-03-01' AND precip_mm IS NOT NULL GROUP BY 1, 2""").toPandas()
inmet["observation_time"] = pd.to_datetime(inmet["observation_time"])

base = {}
for st in est.station_id:
    X = features_mos(serie(st), coluna_previous_runs(LEAD))
    y = inmet[inmet.station_id == st].set_index("observation_time")["chuva"].reindex(X.index)
    X["y"] = (y > 0).astype(float).where(y.notna())
    base[st] = X.dropna(subset=["y", "p_best_match"]).assign(y=lambda d: d["y"].astype(int))
    print(f"{st}: {len(base[st]):,} h ({base[st].index.min():%m/%Y}–{base[st].index.max():%m/%Y}), "
          f"chuva medida em {base[st]['y'].mean():.1%}")

# COMMAND ----------

# Avaliação: cada estação fora do treino, testada nos últimos 12 meses com rótulo
def limiar(y, p):
    return float(max(LIMIARES, key=lambda t: f1_score(y, (p >= t).astype(int), zero_division=0)))

corte = max(b.index.max() for b in base.values()) - pd.DateOffset(months=12)
ys, ps, preds, apis, limiares = [], [], [], [], []
for st in base:
    tr = pd.concat([b[b.index < corte] for k, b in base.items() if k != st])
    grupos = np.concatenate([[k] * int((b.index < corte).sum()) for k, b in base.items() if k != st])
    te = base[st][base[st].index >= corte]
    if len(te) < 500:
        print(f"{st}: só {len(te)} h no teste — fora da avaliação"); continue
    oof = cross_val_predict(HistGradientBoostingClassifier(**HIPER), tr[MOS_FEATURES], tr["y"], groups=grupos,
                            cv=GroupKFold(n_splits=min(3, len(set(grupos)))), method="predict_proba")[:, 1]
    t = limiar(tr["y"].values, oof)
    p = HistGradientBoostingClassifier(**HIPER).fit(tr[MOS_FEATURES], tr["y"]).predict_proba(te[MOS_FEATURES])[:, 1]
    ys.append(te["y"].values); ps.append(p); preds.append((p >= t).astype(int))
    apis.append((te["p_best_match"] > 0).astype(int).values); limiares.append(t)

y_all, p_all, yp_all, api_all = map(np.concatenate, (ys, ps, preds, apis))
threshold = float(np.median(limiares))
met = dict(f1=f1_score(y_all, yp_all), precision=precision_score(y_all, yp_all), recall=recall_score(y_all, yp_all),
           accuracy=accuracy_score(y_all, yp_all), auc_roc=roc_auc_score(y_all, p_all),
           brier=brier_score_loss(y_all, p_all), f1_api=f1_score(y_all, api_all),
           test_size=int(len(y_all)), test_rain_rate=float(y_all.mean()))
print(f"avaliação (estação fora, últimos 12 meses): "
      f"{ {k: round(v, 4) for k, v in met.items()} } | limiares {limiares} → {threshold}")

# COMMAND ----------

# Modelo final com todas as estações e todo o período; registro e alias
df = pd.concat(base.values())
with mlflow.start_run(run_name="MOS_Multimodelo") as run:
    final = HistGradientBoostingClassifier(**HIPER).fit(df[MOS_FEATURES], df["y"])
    mlflow.log_params({"model": "HistGradientBoostingClassifier", "type": "mos_multimodelo",
                       "rotulo": "chuva medida INMET (> 0 na hora)", "features": "rodada de 1 dia antes",
                       "modelos": ",".join(MOS_MODELOS), "estacoes": ",".join(base), "train_size": len(df),
                       "threshold": threshold, **HIPER})
    mlflow.log_metrics({k: v for k, v in met.items() if k != "test_size"})
    X_ex = df[MOS_FEATURES].head(100)
    info = mlflow.sklearn.log_model(final, "model",
                                    signature=infer_signature(X_ex, final.predict_proba(X_ex)[:, 1]))
    run_id = run.info.run_id

client = mlflow.tracking.MlflowClient()
mv = mlflow.register_model(info.model_uri, MODEL_NAME)
for k, v in {"run_id": run_id, "threshold": f"{threshold:.3f}", "f1_estacao_fora": f"{met['f1']:.4f}",
             "f1_api": f"{met['f1_api']:.4f}", "estacoes": ",".join(base)}.items():
    client.set_model_version_tag(MODEL_NAME, mv.version, k, v)

try:
    atual = client.get_model_version_by_alias(MODEL_NAME, "champion")
    f1_atual = float(atual.tags.get("f1_estacao_fora", "0"))
except MlflowException as e:
    if e.error_code != "RESOURCE_DOES_NOT_EXIST":
        raise
    atual, f1_atual = None, None

bate_api = met["f1"] > met["f1_api"]
promote = bate_api and (atual is None or met["f1"] >= f1_atual - 0.02)
if promote:
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
motivo = (("não bate a API: " if not bate_api else "") + f"F1 {met['f1']:.4f} (API {met['f1_api']:.4f})"
          + ("" if atual is None else f" vs champion v{atual.version} {f1_atual:.4f} (tolerância 0,02)"))

spark.createDataFrame([(int(mv.version), run_id, datetime.now(), bool(promote), threshold,
                        float(met["f1"]), float(met["f1_api"]), float(met["precision"]), float(met["recall"]),
                        float(met["accuracy"]), float(met["auc_roc"]), float(met["brier"]), int(len(df)),
                        met["test_size"], met["test_rain_rate"], ",".join(base))],
    "version int, run_id string, created_at timestamp, promoted boolean, threshold double, "
    "f1 double, f1_api double, `precision` double, recall double, accuracy double, auc_roc double, brier double, "
    "train_size int, test_size int, test_rain_rate double, estacoes string"
).write.mode("append").saveAsTable(VERSIONS_TABLE)

print(f"{MODEL_NAME} v{mv.version} | limiar {threshold:.3f} | champion {'PROMOVIDO' if promote else 'mantido'} — {motivo}")
dbutils.notebook.exit(json.dumps({"version": int(mv.version), "promoted": promote, "threshold": threshold,
                                  "metricas": {k: round(v, 4) for k, v in met.items()}, "motivo": motivo}))
