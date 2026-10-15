# Databricks notebook source
# Validação com chuva MEDIDA (INMET) nas cidades das estações automáticas
#
# (a) Quanto o Open-Meteo/ERA5 (best_match) erra? Série horária no ponto exato
#     de cada estação × pluviômetro do INMET, na mesma hora.
# (b) Como o champion se sai fora de Birigui e contra chuva medida? O modelo
#     (treinado em Birigui) recebe as features do Open-Meteo no ponto da
#     estação e prevê "chove na próxima hora"; comparamos com o rótulo do
#     Open-Meteo (como no treino) e com o INMET, nos últimos 12 meses com dado.
#
# Os dados do Open-Meteo por estação ficam em bronze.station_points (base para
# a expansão para outras cidades).

# COMMAND ----------

# MAGIC %run ./features

# COMMAND ----------

import json, time
from datetime import date, timedelta, datetime
import numpy as np
import pandas as pd
import requests
import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
UC_MODEL_NAME = "workspace.default.rain-forecast-birigui"
HOURLY_VARS = ["temperature_2m", "apparent_temperature", "precipitation", "rain",
               "relative_humidity_2m", "windspeed_10m", "dew_point_2m", "pressure_msl",
               "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
               "wind_direction_10m", "total_column_integrated_water_vapour", "boundary_layer_height"]
ANO_INI = 2017                       # Tupã começa em 2017; best_match já é IFS desde 2018
spark.sql("CREATE VOLUME IF NOT EXISTS weather_pipeline.bronze.station_points")
RAW = "/Volumes/weather_pipeline/bronze/station_points"
RESULTS_TABLE = "weather_pipeline.gold.ml_experiments"
ERRO_TABLE = "weather_pipeline.gold.era5_vs_inmet"

est = spark.sql("""
    SELECT station_id, first(station_name) AS nome, first(lat) AS lat, first(lon) AS lon,
           first(dist_km) AS dist_km, max(observation_time) AS fim_inmet
    FROM weather_pipeline.silver.inmet_hourly GROUP BY station_id ORDER BY dist_km
""").toPandas()
print(est.to_string(index=False))

# COMMAND ----------

# Open-Meteo no ponto de cada estação (um arquivo por estação/ano; anos fechados são reaproveitados)
def fetch(lat, lon, start, end, tentativas=8):
    for i in range(tentativas):
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
            "hourly": ",".join(HOURLY_VARS), "timezone": "America/Sao_Paulo",
            "models": "best_match"}, timeout=180)
        if r.status_code == 429 or r.status_code >= 500:   # limite da API ou falha temporária
            time.sleep(30 * (i + 1)); continue
        r.raise_for_status()
        return r.json()["hourly"]
    raise RuntimeError(f"Archive recusou ({lat},{lon}) {start}..{end}")

fim = date.today() - timedelta(days=5)
existentes = {f.name for f in dbutils.fs.ls(RAW)}
for _, e in est.iterrows():
    for ano in range(ANO_INI, fim.year + 1):
        nome = f"{e.station_id}_{ano}.json"
        if nome in existentes and ano < fim.year:
            continue
        h = fetch(e.lat, e.lon, f"{ano}-01-01", min(date(ano, 12, 31), fim).isoformat())
        dbutils.fs.put(f"{RAW}/{nome}", json.dumps(h), overwrite=True)
        time.sleep(4)

# COMMAND ----------

def serie_om(cod):
    fr = [pd.DataFrame(json.loads(dbutils.fs.head(f.path, 100_000_000)))
          for f in dbutils.fs.ls(RAW) if f.name.startswith(f"{cod}_")]
    d = pd.concat(fr, ignore_index=True).drop_duplicates("time")
    d["observation_time"] = pd.to_datetime(d.pop("time"))
    return d.rename(columns={"relative_humidity_2m": "relative_humidity", "windspeed_10m": "windspeed",
                             "total_column_integrated_water_vapour": "tcwv"})

# features da variante B em pandas: add_features_pandas() de ./features

inmet = spark.table("weather_pipeline.silver.inmet_hourly").select(
    "station_id", "observation_time", "precip_mm").toPandas()
inmet["observation_time"] = pd.to_datetime(inmet["observation_time"])
# horário de verão (até 2019) gera horas locais repetidas: soma a chuva da hora duplicada
inmet = inmet.groupby(["station_id", "observation_time"], as_index=False)["precip_mm"].sum(min_count=1)

# COMMAND ----------

# (a) Open-Meteo × INMET na mesma hora
def tab(om, ob, lim):
    a, b = om >= lim if lim > 0 else om > 0, ob >= lim if lim > 0 else ob > 0
    vp, fp, fn, vn = (a & b).sum(), (a & ~b).sum(), (~a & b).sum(), (~a & ~b).sum()
    return dict(limiar_mm=lim, horas=int(len(a)), freq_om=float(a.mean()), freq_inmet=float(b.mean()),
                pod=float(vp / max(vp + fn, 1)), far=float(fp / max(vp + fp, 1)),
                csi=float(vp / max(vp + fp + fn, 1)), vies_freq=float((vp + fp) / max(vp + fn, 1)))

erros, series = [], {}
for _, e in est.iterrows():
    s = add_features_pandas(serie_om(e.station_id))
    m = inmet[inmet.station_id == e.station_id].set_index("observation_time")["precip_mm"]
    s["inmet_mm"] = m.reindex(s.index)
    series[e.station_id] = s
    ok = s[["precipitation", "inmet_mm"]].dropna()
    for lim in (0.0, 0.5, 1.0):
        erros.append(dict(station_id=e.station_id, nome=e.nome, dist_km=float(e.dist_km),
                          **tab(ok["precipitation"].values, ok["inmet_mm"].values, lim)))
    mens = ok.resample("MS").sum(min_count=600).dropna()   # meses com dado quase completo
    erros[-1]["corr_mensal"] = float(mens.corr().iloc[0, 1]) if len(mens) > 6 else None
    erros[-1]["vies_volume"] = float(mens["precipitation"].sum() / mens["inmet_mm"].sum() - 1) if len(mens) else None

err = pd.DataFrame(erros)
spark.createDataFrame(err.astype(object).where(err.notna(), None)) \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(ERRO_TABLE)
print(err.round(3).to_string(index=False))

# COMMAND ----------

# (b) champion nas cidades das estações — últimos 12 meses com dado do INMET
client = mlflow.tracking.MlflowClient()
mv = client.get_model_version_by_alias(UC_MODEL_NAME, "champion")
model = mlflow.sklearn.load_model(f"models:/{UC_MODEL_NAME}@champion")
thr = float(mv.tags["threshold"])
feats = list(model.feature_names_in_)

resultados = []
for _, e in est.iterrows():
    s = series[e.station_id]
    s["y_om"] = (s["precipitation"].shift(-1) > 0).astype(float).where(s["precipitation"].shift(-1).notna())
    s["y_inmet"] = (s["inmet_mm"].shift(-1) > 0).astype(float).where(s["inmet_mm"].shift(-1).notna())
    s["persist_inmet"] = (s["inmet_mm"] > 0).astype(float).where(s["inmet_mm"].notna())
    fim_e = pd.Timestamp(e.fim_inmet)
    jan = s[(s.index > fim_e - pd.DateOffset(months=12)) & (s.index <= fim_e)]
    jan = jan.dropna(subset=feats + ["y_om", "y_inmet", "persist_inmet"])
    if len(jan) < 500:
        print(e.station_id, "pouco dado no período:", len(jan)); continue
    prob = model.predict_proba(jan[feats])[:, 1]; pred = (prob >= thr).astype(int)
    for alvo, y, yp, score in (
        ("rotulo_open_meteo", jan["y_om"], pred, prob),
        ("chuva_medida_inmet", jan["y_inmet"], pred, prob),
        # referência ingênua: "se está chovendo agora (INMET), vai chover na próxima hora"
        ("persistencia_inmet__vs_inmet", jan["y_inmet"], jan["persist_inmet"].astype(int), None)):
        y = y.astype(int)
        r = dict(experimento="inmet_validacao", variante=f"{e.station_id}_{alvo}", n_features=len(feats),
                 threshold=thr if score is not None else None, f1=f1_score(y, yp),
                 auc_roc=roc_auc_score(y, score) if score is not None else None,
                 accuracy=accuracy_score(y, yp), precision=precision_score(y, yp, zero_division=0),
                 recall=recall_score(y, yp), train_size=None, test_size=int(len(y)))
        resultados.append(r)

f = lambda v: None if v is None else float(v)
agora = datetime.now()
spark.createDataFrame(
    [(r["experimento"], r["variante"], r["n_features"], f(r["threshold"]), f(r["f1"]), f(r["auc_roc"]),
      f(r["accuracy"]), f(r["precision"]), f(r["recall"]), r["train_size"], r["test_size"], agora) for r in resultados],
    "experimento string, variante string, n_features bigint, threshold double, f1 double, auc_roc double, "
    "accuracy double, `precision` double, recall double, train_size bigint, test_size bigint, created_at timestamp"
).write.mode("append").saveAsTable(RESULTS_TABLE)

dbutils.notebook.exit(json.dumps({
    "modelo": f"v{mv.version} limiar {thr}",
    "erro_open_meteo_vs_inmet": err.round(3).to_dict("records"),
    "validacao": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                   if k in ("variante", "f1", "precision", "recall", "accuracy", "auc_roc", "test_size")}
                  for r in resultados]}, default=str))
