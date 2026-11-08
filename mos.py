# Databricks notebook source
# MOS multimodelo — definição única das features, usada via `%run ./mos` pelo
# 16_train_mos_multimodelo (treino com a Previous Runs API) e pelo
# 17_inference_mos_multimodelo (produção com a Forecast API).
#
# São as features do experimento 14 (o notebook do experimento fica como
# registro e não usa este arquivo). A única diferença entre treino e produção é
# o nome das colunas que chegam da Open-Meteo:
#   Previous Runs: precipitation_previous_day1_ecmwf_ifs025
#   Forecast     : precipitation_ecmwf_ifs025

# COMMAND ----------

import numpy as np
import pandas as pd

MOS_MODELOS = ["best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "jma_seamless", "gem_seamless"]
MOS_VARS = ["precipitation", "cloud_cover", "relative_humidity_2m", "cape"]
MOS_CAPE = ("best_match", "gfs_seamless", "gem_seamless")   # os que têm CAPE nas rodadas anteriores


def coluna_previous_runs(lead):
    return lambda v, m: f"{v}_previous_day{lead}_{m}"


def coluna_forecast(v, m):
    return f"{v}_{m}"


def features_mos(s, coluna):
    """Features da hora t. `s`: DataFrame horário contínuo (índice = hora local)
    com as colunas da Open-Meteo; `coluna(v, m)` dá o nome da variável v do modelo m."""
    X = pd.DataFrame(index=s.index)
    p = pd.DataFrame({m: s.get(coluna("precipitation", m)) for m in MOS_MODELOS}, index=s.index).astype(float)
    for m in MOS_MODELOS:
        X[f"p_{m}"] = p[m]
        X[f"p3_{m}"] = p[m].rolling(3, center=True, min_periods=1).sum()   # t−1, t, t+1
        X[f"nuv_{m}"] = s.get(coluna("cloud_cover", m))
        X[f"ur_{m}"] = s.get(coluna("relative_humidity_2m", m))
    for m in MOS_CAPE:
        X[f"cape_{m}"] = s.get(coluna("cape", m))
    X["ens_media"], X["ens_max"] = p.mean(axis=1), p.max(axis=1)
    X["ens_frac"] = (p >= 0.1).mean(axis=1).where(p.notna().any(axis=1))
    X["ens_frac3"] = (p.rolling(3, center=True, min_periods=1).sum() >= 0.1).mean(axis=1)
    h, mo = X.index.hour, X.index.month
    X["hora_sin"], X["hora_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    X["mes_sin"], X["mes_cos"] = np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12)
    return X.astype(float)


MOS_FEATURES = list(features_mos(pd.DataFrame(index=pd.date_range("2024-01-01", periods=3, freq="h")),
                                 coluna_forecast).columns)
print(f"mos carregado — {len(MOS_FEATURES)} features, {len(MOS_MODELOS)} modelos")
