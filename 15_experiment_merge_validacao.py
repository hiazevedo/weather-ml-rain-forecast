# Databricks notebook source
# Experimento 15 — o MERGE/CPTEC serve de "chuva medida" para Birigui?
#
# Birigui não tem pluviômetro automático. O MERGE (INPE) junta satélite
# (GPM-IMERG) e pluviômetros numa grade de 10 km, hora a hora
# (silver.merge_hourly, notebook 15_collect_merge do weather-dlt-pipeline).
# Antes de usá-lo como rótulo, comparamos com chuva medida de verdade:
#
#   1) INMET, hora a hora, nos pontos das 4 estações — define o alinhamento da
#      hora (defasagem). O MERGE usa essas estações, então é o melhor caso.
#   2) Os 5 postos da ANA em Birigui (≤ 6 km), leitura diária às 7h, jun/2023 a
#      jan/2024 — independentes do MERGE. O Open-Meteo (Archive) entra ao lado
#      para comparar: qual dos dois chega mais perto dos pluviômetros?
#   3) Fração de horas com chuva: MERGE × Open-Meteo × INMET.
#
# Resultado em gold.merge_validacao (uma linha por métrica).

# COMMAND ----------

from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

RESULT = "weather_pipeline.gold.merge_validacao"
POSTOS_BIRIGUI = ["02150015", "02150095", "02150094", "02150096", "02150016"]
CHUVA_H = 0.2    # mm/h: resolução do pluviômetro automático
CHUVA_D = 1.0    # mm/dia: dia com chuva (OMM)

merge = spark.table("weather_pipeline.silver.merge_hourly").select(
    "ponto", "observation_time", "prec_mm").toPandas()
merge["observation_time"] = pd.to_datetime(merge["observation_time"])
inmet = spark.sql("""
    SELECT station_id AS ponto, observation_time, sum(precip_mm) AS chuva
    FROM weather_pipeline.silver.inmet_hourly WHERE precip_mm IS NOT NULL
    GROUP BY 1, 2""").toPandas()
inmet["observation_time"] = pd.to_datetime(inmet["observation_time"])
om = spark.sql("""
    SELECT observation_time, precipitation AS chuva FROM weather_pipeline.silver.weather_clean
    WHERE _source_file LIKE '%historical%' AND observation_time >= '2023-05-01'""").toPandas()
om["observation_time"] = pd.to_datetime(om["observation_time"])
ana = spark.sql(f"""
    SELECT date, avg(chuva_mm) AS chuva, count(chuva_mm) AS postos
    FROM weather_pipeline.silver.ana_diaria
    WHERE station_id IN ({",".join(f"'{p}'" for p in POSTOS_BIRIGUI)})
      AND date >= '2023-06-01' AND chuva_mm IS NOT NULL
    GROUP BY date""").toPandas()
ana["date"] = pd.to_datetime(ana["date"])
print(f"MERGE: {merge.observation_time.min()} → {merge.observation_time.max()}, pontos {sorted(merge.ponto.unique())}")
linhas = []
def guarda(metrica, fonte, ponto, valor, n):
    linhas.append((metrica, fonte, ponto, None if valor is None or pd.isna(valor) else float(valor), int(n), datetime.now()))

# COMMAND ----------

# 1) INMET hora a hora: qual defasagem alinha o MERGE à hora local da silver?
res = []
for lag in (-2, -1, 0, 1, 2):
    ys, ms = [], []
    for st in inmet.ponto.unique():
        m = merge[merge.ponto == st].set_index("observation_time")["prec_mm"]
        m.index = m.index + pd.Timedelta(hours=lag)
        i = inmet[inmet.ponto == st].set_index("observation_time")["chuva"]
        j = pd.concat([i.rename("inmet"), m.rename("merge")], axis=1, join="inner").dropna()
        ys.append(j["inmet"]); ms.append(j["merge"])
    y, m = pd.concat(ys), pd.concat(ms)
    res.append((lag, len(y), y.corr(m), f1_score(y >= CHUVA_H, m >= CHUVA_H)))
lags = pd.DataFrame(res, columns=["defasagem_h", "horas", "corr", "f1"])
print(lags.to_string(index=False))
LAG = int(lags.loc[lags["corr"].idxmax(), "defasagem_h"])
print(f"defasagem escolhida: {LAG:+d} h")
for r in lags.itertuples():
    guarda(f"inmet_horario_corr_defasagem_{r.defasagem_h:+d}", "merge", "4 estações INMET", r.corr, r.horas)
    guarda(f"inmet_horario_f1_defasagem_{r.defasagem_h:+d}", "merge", "4 estações INMET", r.f1, r.horas)

merge["observation_time"] = merge["observation_time"] + pd.Timedelta(hours=LAG)

# COMMAND ----------

# 2) Postos da ANA em Birigui (diário, leitura às 7h) × MERGE × Open-Meteo
def diario(serie, desloc_dia):
    """Soma das horas 07h(d−1)–06h(d) (desloc 0) ou 07h(d)–06h(d+1) (desloc 1)."""
    s = serie.copy()
    s.index = (s.index - pd.Timedelta(hours=7)).normalize() + pd.Timedelta(days=1 - desloc_dia)
    return s.groupby(level=0).sum(min_count=20)   # dia com ≥ 20 h válidas

m_bir = merge[merge.ponto == "birigui"].set_index("observation_time")["prec_mm"]
o_bir = om.set_index("observation_time")["chuva"]
a = ana.set_index("date")["chuva"]
melhor = None
for desloc in (0, 1):
    j = pd.concat([a.rename("ana"), diario(m_bir, desloc).rename("merge")], axis=1, join="inner").dropna()
    c = j["ana"].corr(j["merge"])
    print(f"ANA × MERGE, convenção do dia {desloc}: corr {c:.3f} ({len(j)} dias)")
    if melhor is None or c > melhor[1]:
        melhor = (desloc, c)
DESLOC = melhor[0]

j = pd.concat([a.rename("ana"), diario(m_bir, DESLOC).rename("merge"),
               diario(o_bir, DESLOC).rename("openmeteo")], axis=1, join="inner").dropna()
print(f"\nBirigui, {len(j)} dias ({j.index.min():%d/%m/%Y}–{j.index.max():%d/%m/%Y}), média dos postos da ANA")
print(f"{'':<10}{'corr':>6}{'total/ANA':>11}{'dias c/ chuva':>15}{'F1 dia chuva':>14}")
print(f"{'ANA':<10}{'':>6}{1:>11.2f}{(j['ana'] >= CHUVA_D).mean():>15.1%}{'':>14}")
for f in ("merge", "openmeteo"):
    corr, razao = j["ana"].corr(j[f]), j[f].sum() / j["ana"].sum()
    freq, f1 = (j[f] >= CHUVA_D).mean(), f1_score(j["ana"] >= CHUVA_D, j[f] >= CHUVA_D)
    print(f"{f:<10}{corr:>6.3f}{razao:>11.2f}{freq:>15.1%}{f1:>14.3f}")
    guarda("ana_birigui_diario_corr", f, "birigui", corr, len(j))
    guarda("ana_birigui_diario_total_razao", f, "birigui", razao, len(j))
    guarda("ana_birigui_dias_chuva_f1", f, "birigui", f1, len(j))
    guarda("ana_birigui_dias_chuva_frac", f, "birigui", freq, len(j))
guarda("ana_birigui_dias_chuva_frac", "ana", "birigui", (j["ana"] >= CHUVA_D).mean(), len(j))

# COMMAND ----------

# 3) Fração de horas com chuva (≥ 0,2 mm), mesmas horas
ult = merge.observation_time.max() - pd.DateOffset(months=12)
jb = pd.concat([m_bir.rename("merge"), o_bir.rename("openmeteo")], axis=1, join="inner").dropna()
jb = jb[jb.index >= ult]
print(f"\nBirigui, últimos 12 meses ({len(jb):,} h): horas com chuva — "
      f"MERGE {(jb['merge'] >= CHUVA_H).mean():.1%} · Open-Meteo {(jb['openmeteo'] >= CHUVA_H).mean():.1%} "
      f"· Open-Meteo > 0 {(jb['openmeteo'] > 0).mean():.1%}")
guarda("horas_chuva_frac_12m", "merge", "birigui", (jb["merge"] >= CHUVA_H).mean(), len(jb))
guarda("horas_chuva_frac_12m", "openmeteo", "birigui", (jb["openmeteo"] >= CHUVA_H).mean(), len(jb))
guarda("horas_chuva_frac_12m_maior_que_0", "openmeteo", "birigui", (jb["openmeteo"] > 0).mean(), len(jb))

js = inmet.merge(merge, on=["ponto", "observation_time"], how="inner").dropna()
print(f"Estações do INMET ({len(js):,} h): horas com chuva — INMET {(js['chuva'] >= CHUVA_H).mean():.1%} "
      f"· MERGE {(js['prec_mm'] >= CHUVA_H).mean():.1%}")
guarda("horas_chuva_frac", "inmet", "4 estações INMET", (js["chuva"] >= CHUVA_H).mean(), len(js))
guarda("horas_chuva_frac", "merge", "4 estações INMET", (js["prec_mm"] >= CHUVA_H).mean(), len(js))

spark.createDataFrame(linhas, "metrica string, fonte string, ponto string, valor double, n bigint, created_at timestamp"
                      ).write.mode("append").saveAsTable(RESULT)
print(f"\n{len(linhas)} métricas gravadas em {RESULT} (defasagem {LAG:+d} h, convenção do dia ANA {DESLOC})")
