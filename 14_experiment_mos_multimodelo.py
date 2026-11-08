# Databricks notebook source
# Experimento 14 — MOS multimodelo
#
# Pergunta: combinar as previsões de 6 modelos (ECMWF IFS, GFS, ICON, JMA, GEM e o
# best_match da Open-Meteo), feitas 1 e 2 dias antes, prevê "vai chover na hora"
# melhor do que a previsão crua da Open-Meteo (best_match > 0 mm)?
#
# Motivo: em produção o champion não supera a própria API (F1 0,630 contra 0,655
# a 1–6 h, de 15/10 a 07/11/2026), e o experimento 09 (MOS só com o best_match)
# empatou com ela. Um modelo só repete a API; vários modelos trazem informação nova.
#
# Dois rótulos, nas MESMAS horas para todas as abordagens:
#   A) Birigui — chuva > 0 no Archive (rótulo Open-Meteo, o mesmo do champion)
#   B) 4 estações do INMET — chuva MEDIDA > 0, deixando cada estação de fora do
#      treino (como Birigui, que não tem estação)
#
# Abordagens comparadas
#   api       — best_match > 0 mm (é o que o painel mostra)
#   melhor_1  — o melhor modelo sozinho (> 0 mm), escolhido no treino
#   votos     — fração de modelos com chuva ≥ 0,1 mm, limiar escolhido no treino
#   mos       — HistGradientBoosting com os 6 modelos (chuva na hora e vizinhas,
#               nuvens, umidade, CAPE, consenso) + hora/mês; limiar pelo F1 fora da dobra
#
# Dados: Previous Runs API (rodadas de 1 e 2 dias antes, desde mar/2024), cache
# mensal no Volume bronze.previous_runs_multi. Nada aqui muda produção.

# COMMAND ----------

import json, time
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             roc_auc_score, brier_score_loss)

MODELOS = ["best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "jma_seamless", "gem_seamless"]
VARS = ["precipitation", "cloud_cover", "relative_humidity_2m", "cape"]
LEADS = [1, 2]
INICIO = date(2024, 3, 1)
FIM = date.today() - timedelta(days=5)          # o Archive (rótulo de Birigui) atrasa ~5 dias
spark.sql("CREATE VOLUME IF NOT EXISTS weather_pipeline.bronze.previous_runs_multi")
RAW = "/Volumes/weather_pipeline/bronze/previous_runs_multi"
RESULTS_TABLE = "weather_pipeline.gold.ml_experiments"
EXPERIMENTO = "mos_multimodelo"

est = spark.sql("""
    SELECT station_id, first(lat) AS lat, first(lon) AS lon, first(dist_km) AS dist_km
    FROM weather_pipeline.silver.inmet_hourly GROUP BY station_id ORDER BY dist_km
""").toPandas()
PONTOS = {"birigui": (-21.2878, -50.3409), **{r.station_id: (r.lat, r.lon) for r in est.itertuples()}}
print(PONTOS)

# COMMAND ----------

# Previous Runs: um arquivo por ponto e mês; meses fechados já baixados são reaproveitados
hourly = [f"{v}_previous_day{l}" for v in VARS for l in LEADS]

def fetch(lat, lon, ini, fim, tentativas=8):
    for i in range(tentativas):
        r = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "start_date": ini, "end_date": fim,
            "hourly": ",".join(hourly), "models": ",".join(MODELOS),
            "timezone": "America/Sao_Paulo"}, timeout=180)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(30 * (i + 1)); continue
        r.raise_for_status()
        return r.json()["hourly"]
    raise RuntimeError(f"Previous Runs recusou ({lat},{lon}) {ini}..{fim}")

# O IP de saída do Databricks é compartilhado e a API às vezes recusa (429) por
# horas; nesse caso, baixe os meses pela rede de casa e envie ao Volume (mesmo
# nome de arquivo) — o notebook reaproveita. Arquivos do mês corrente com menos
# de 24 h também são reaproveitados.
existentes = {f.name: f.modificationTime / 1000 for f in dbutils.fs.ls(RAW)}
for ponto, (lat, lon) in PONTOS.items():
    mes = INICIO
    while mes <= FIM:
        prox = (mes.replace(day=28) + timedelta(days=4)).replace(day=1)
        nome = f"{ponto}_{mes:%Y_%m}.json"
        recente = time.time() - existentes.get(nome, 0) < 24 * 3600
        if not (nome in existentes and (prox <= FIM or recente)):
            h = fetch(lat, lon, mes.isoformat(), min(prox - timedelta(days=1), FIM).isoformat())
            dbutils.fs.put(f"{RAW}/{nome}", json.dumps(h), overwrite=True)
            time.sleep(3)
        mes = prox

def serie(ponto):
    fr = [pd.DataFrame(json.loads(dbutils.fs.head(f.path, 100_000_000)))
          for f in dbutils.fs.ls(RAW) if f.name.startswith(f"{ponto}_")]
    d = pd.concat(fr, ignore_index=True).drop_duplicates("time")
    d["observation_time"] = pd.to_datetime(d.pop("time"))
    return d.set_index("observation_time").sort_index()

# COMMAND ----------

# Rótulos: Archive de Birigui (silver) e chuva medida do INMET
obs_bir = spark.sql("""
    SELECT observation_time, precipitation AS chuva FROM weather_pipeline.silver.weather_clean
    WHERE observation_time >= '2024-03-01' AND _source_file LIKE '%historical%'
""").toPandas()
obs_bir["observation_time"] = pd.to_datetime(obs_bir["observation_time"])
inmet = spark.sql("""
    SELECT station_id, observation_time, sum(precip_mm) AS chuva
    FROM weather_pipeline.silver.inmet_hourly
    WHERE observation_time >= '2024-03-01' AND precip_mm IS NOT NULL
    GROUP BY station_id, observation_time
""").toPandas()
inmet["observation_time"] = pd.to_datetime(inmet["observation_time"])

def features(s, lead):
    """Features da hora t a partir das rodadas de `lead` dias antes."""
    X = pd.DataFrame(index=s.index)
    p = pd.DataFrame({m: s.get(f"precipitation_previous_day{lead}_{m}") for m in MODELOS}).astype(float)
    for m in MODELOS:
        X[f"p_{m}"] = p[m]
        X[f"p3_{m}"] = p[m].rolling(3, center=True, min_periods=1).sum()   # t−1, t, t+1
        X[f"nuv_{m}"] = s.get(f"cloud_cover_previous_day{lead}_{m}")
        X[f"ur_{m}"] = s.get(f"relative_humidity_2m_previous_day{lead}_{m}")
    for m in ("best_match", "gfs_seamless", "gem_seamless"):
        X[f"cape_{m}"] = s.get(f"cape_previous_day{lead}_{m}")
    X["ens_media"], X["ens_max"] = p.mean(axis=1), p.max(axis=1)
    X["ens_frac"] = (p >= 0.1).mean(axis=1).where(p.notna().any(axis=1))
    X["ens_frac3"] = (p.rolling(3, center=True, min_periods=1).sum() >= 0.1).mean(axis=1)
    h, mo = X.index.hour, X.index.month
    X["hora_sin"], X["hora_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    X["mes_sin"], X["mes_cos"] = np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12)
    return X.astype(float)

# COMMAND ----------

LIMIARES = np.round(np.arange(0.05, 0.96, 0.05), 2)

def melhor_limiar(y, score, candidatos=LIMIARES):
    return float(max(candidatos, key=lambda t: f1_score(y, (score >= t).astype(int), zero_division=0)))

def metricas(y, pred, score=None):
    m = dict(f1=f1_score(y, pred, zero_division=0), precision=precision_score(y, pred, zero_division=0),
             recall=recall_score(y, pred, zero_division=0), accuracy=accuracy_score(y, pred))
    m["auc_roc"] = roc_auc_score(y, score) if score is not None and len(set(y)) > 1 else None
    return m

def novo_mos():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=42)

def avaliar(tr, te, grupos_tr, rotulo, lead):
    """Ajusta as 4 abordagens no treino e mede no teste (mesmas horas)."""
    feats = [c for c in tr.columns if c not in ("y",)]
    ytr, yte = tr["y"].values, te["y"].values
    out = []
    # api: best_match > 0
    out.append(("api", 0.0, metricas(yte, (te["p_best_match"] > 0).astype(int), te["p_best_match"].fillna(0)), 1))
    # melhor modelo sozinho, escolhido pelo F1 no treino
    f1s = {m: f1_score(ytr, (tr[f"p_{m}"] > 0).astype(int)) for m in MODELOS}
    m1 = max(f1s, key=f1s.get)
    out.append((f"melhor_1:{m1}", 0.0, metricas(yte, (te[f"p_{m1}"] > 0).astype(int), te[f"p_{m1}"].fillna(0)), 1))
    # votos
    kv = melhor_limiar(ytr, tr["ens_frac"].fillna(0), np.round(np.arange(1, 7) / 6, 3))
    out.append(("votos", kv, metricas(yte, (te["ens_frac"].fillna(0) >= kv).astype(int), te["ens_frac"].fillna(0)), 1))
    # MOS: limiar pelo F1 nas previsões fora da dobra do treino
    oof = cross_val_predict(novo_mos(), tr[feats], ytr, groups=grupos_tr,
                            cv=GroupKFold(n_splits=min(4, len(set(grupos_tr)))), method="predict_proba")[:, 1]
    thr = melhor_limiar(ytr, oof)
    mos = novo_mos().fit(tr[feats], ytr)
    p = mos.predict_proba(te[feats])[:, 1]
    met = metricas(yte, (p >= thr).astype(int), p)
    met["brier"] = brier_score_loss(yte, p)
    out.append(("mos", thr, met, len(feats)))
    linhas = []
    for nome, limiar, met, nf in out:
        print(f"  {rotulo} d{lead} {nome:<28} F1 {met['f1']:.3f}  prec {met['precision']:.3f}  "
              f"rec {met['recall']:.3f}  AUC {met['auc_roc'] or float('nan'):.3f}"
              + (f"  Brier {met['brier']:.3f}" if "brier" in met else "") + f"  (limiar {limiar})")
        linhas.append((EXPERIMENTO, f"{rotulo}_d{lead}_{nome}", nf, float(limiar), float(met["f1"]),
                       None if met["auc_roc"] is None else float(met["auc_roc"]), float(met["accuracy"]),
                       float(met["precision"]), float(met["recall"]), int(len(tr)), int(len(te)), datetime.now()))
    return linhas

# COMMAND ----------

# A) Birigui, rótulo Open-Meteo (Archive): teste = últimos 12 meses com rótulo
resultados = []
s_bir = serie("birigui")
y_bir = obs_bir.set_index("observation_time")["chuva"]
for lead in LEADS:
    X = features(s_bir, lead)
    X["y"] = (y_bir.reindex(X.index) > 0).astype(float).where(y_bir.reindex(X.index).notna())
    X = X.dropna(subset=["y", "p_best_match"])
    X["y"] = X["y"].astype(int)
    corte = X.index.max() - pd.DateOffset(months=12)
    tr, te = X[X.index < corte], X[X.index >= corte]
    print(f"Birigui d{lead}: treino {len(tr):,} h ({tr.index.min():%m/%Y}–{tr.index.max():%m/%Y}), "
          f"teste {len(te):,} h; chuva em {te['y'].mean():.1%} das horas do teste")
    resultados += avaliar(tr, te, tr.index.to_period("M").astype(str), "A_birigui_openmeteo", lead)

# COMMAND ----------

# B) Estações do INMET, chuva MEDIDA: cada estação fica fora do treino e é testada
#    nos últimos 12 meses com dado; as métricas juntam as horas de teste das 4.
for lead in LEADS:
    base = {}
    for st in est.station_id:
        X = features(serie(st), lead)
        y = inmet[inmet.station_id == st].set_index("observation_time")["chuva"].reindex(X.index)
        X["y"] = (y > 0).astype(float).where(y.notna())
        base[st] = X.dropna(subset=["y", "p_best_match"]).assign(y=lambda d: d["y"].astype(int))
    fim_rotulo = max(b.index.max() for b in base.values())
    corte = fim_rotulo - pd.DateOffset(months=12)
    por_abordagem = {}
    for st in est.station_id:
        tr = pd.concat([b[b.index < corte] for k, b in base.items() if k != st])
        grupos = np.concatenate([[k] * int((b.index < corte).sum()) for k, b in base.items() if k != st])
        te = base[st][base[st].index >= corte]
        if len(te) < 500:
            print(f"{st}: só {len(te)} h no teste — fora"); continue
        print(f"INMET d{lead}, estação fora = {st}: treino {len(tr):,} h, teste {len(te):,} h")
        for linha in avaliar(tr, te, grupos, f"B_inmet_{st}", lead):
            resultados.append(linha)
            por_abordagem.setdefault(linha[1].split(f"_d{lead}_")[1].split(":")[0], []).append(
                (linha[4], linha[10]))
    print(f"--- INMET d{lead}, F1 médio ponderado pelas horas de teste:")
    for nome, v in por_abordagem.items():
        print(f"  {nome:<10} {sum(f * n for f, n in v) / sum(n for _, n in v):.3f}")

# COMMAND ----------

spark.createDataFrame(resultados, "experimento string, variante string, n_features bigint, threshold double, "
                      "f1 double, auc_roc double, accuracy double, `precision` double, recall double, "
                      "train_size bigint, test_size bigint, created_at timestamp"
                      ).write.mode("append").saveAsTable(RESULTS_TABLE)
print(f"{len(resultados)} linhas gravadas em {RESULTS_TABLE} (experimento = '{EXPERIMENTO}')")
