# Databricks notebook source
# MAGIC %pip install prophet
# MAGIC
# MAGIC print("Reiniciando kernel")
# MAGIC dbutils.library.restartPython()
# MAGIC print("Kernel reiniciado")

# COMMAND ----------

import mlflow
import mlflow.spark
import mlflow.sklearn
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql import functions as F

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              mean_squared_error, r2_score, mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from prophet import Prophet

# COMMAND ----------

# MLflow storage
MLFLOW_TMP = "/Volumes/weather_pipeline/bronze/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP

# Criar volume se não existir
spark.sql("CREATE VOLUME IF NOT EXISTS weather_pipeline.bronze.mlflow_tmp")

EXPERIMENT_NAME = "/Users/{}/weather-ml-rain-forecast/weather-ml-rain-forecast".format(
    spark.sql("SELECT current_user()").collect()[0][0]
)
mlflow.set_experiment(EXPERIMENT_NAME)

FEATURE_TABLE = "weather_pipeline.gold.rain_features"

ML_FEATURES = [
    "temperature_2m", "apparent_temperature",
    "precipitation", "relative_humidity", "windspeed",
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

TARGET_CLF = "target_will_rain"
TARGET_REG = "target_rain_next_1h"

print(f"   MLflow configurado")
print(f"   Experiment : {EXPERIMENT_NAME}")
print(f"   Features   : {len(ML_FEATURES)}")

# COMMAND ----------

# Preparar dados com split temporal (não aleatório!)

print("Preparando dados com split temporal...\n")

# IMPORTANTE: Para séries temporais, split deve ser por data
# Treino: 1940–2022 | Validação: 2023 | Teste: 2024–2026
df = spark.table(FEATURE_TABLE) \
    .dropna(subset=ML_FEATURES + [TARGET_CLF, TARGET_REG])

train_df = df.filter(F.col("observation_time") <  "2023-01-01")
val_df   = df.filter((F.col("observation_time") >= "2023-01-01") &
                     (F.col("observation_time") <  "2024-01-01"))
test_df  = df.filter(F.col("observation_time") >= "2024-01-01")

print(f"   Treino     : {train_df.count():,} (1940–2022)")
print(f"   Validação  : {val_df.count():,}   (2023)")
print(f"   Teste      : {test_df.count():,}   (2024–2026)")

# Converter para Pandas para sklearn
print("\n   Convertendo para Pandas...")
X_train = train_df.select(ML_FEATURES).toPandas()
y_clf_train = train_df.select(TARGET_CLF).toPandas().values.ravel()
y_reg_train = train_df.select(TARGET_REG).toPandas().values.ravel()

X_test  = test_df.select(ML_FEATURES).toPandas()
y_clf_test  = test_df.select(TARGET_CLF).toPandas().values.ravel()
y_reg_test  = test_df.select(TARGET_REG).toPandas().values.ravel()

print(f"   Dados prontos!")
print(f"   Taxa de chuva treino : {y_clf_train.mean()*100:.1f}%")
print(f"   Taxa de chuva teste  : {y_clf_test.mean()*100:.1f}%")

# COMMAND ----------

# RandomForest Regressor

print("Treinando RandomForest...\n")

# Classificador
with mlflow.start_run(run_name="RF_Classifier_v2") as run:
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train, y_clf_train)
    y_pred_rf = rf_clf.predict(X_test)
    y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_clf_test, y_pred_rf)
    f1   = f1_score(y_clf_test, y_pred_rf)
    auc  = roc_auc_score(y_clf_test, y_prob_rf)
    prec = precision_score(y_clf_test, y_pred_rf)
    rec  = recall_score(y_clf_test, y_pred_rf)

    mlflow.log_params({
        "model": "RandomForest", "type": "classifier",
        "n_estimators": 200, "max_depth": 15,
        "class_weight": "balanced", "version": "v2"
    })
    mlflow.log_metrics({
        "accuracy": acc, "f1": f1, "auc_roc": auc,
        "precision": prec, "recall": rec
    })
    mlflow.sklearn.log_model(
        rf_clf, "model",
        input_example=X_train.iloc[:5]
    )
    print(f"   RF Classifier — F1: {f1:.4f} | AUC: {auc:.4f}")

# Regressor
# Estratégia: pipeline de 2 etapas
# Etapa 1: classificador decide SE vai chover
# Etapa 2: regressor decide QUANTO vai chover (só quando etapa 1 = sim)
with mlflow.start_run(run_name="RF_Regressor_v2") as run:
    rf_reg = RandomForestRegressor(
        n_estimators=200, max_depth=15,
        random_state=42, n_jobs=-1
    )

    # Treinar só com horas que choveu
    mask_train = y_reg_train > 0
    rf_reg.fit(X_train[mask_train], y_reg_train[mask_train])

    # Predição em 2 etapas:
    # 1. Classificador decide se vai chover
    pred_vai_chover = rf_clf.predict(X_test)
    # 2. Para quem o classificador disse SIM, regressor prevê quanto
    y_pred_final = np.zeros(len(X_test))
    mask_test_pred = pred_vai_chover == 1
    if mask_test_pred.sum() > 0:
        y_pred_final[mask_test_pred] = np.maximum(
            rf_reg.predict(X_test[mask_test_pred]), 0
        )

    # Avaliar só nas horas que realmente choveu
    mask_test_real = y_reg_test > 0
    rmse_rain = np.sqrt(mean_squared_error(
        y_reg_test[mask_test_real],
        y_pred_final[mask_test_real]
    ))
    mae_rain = mean_absolute_error(
        y_reg_test[mask_test_real],
        y_pred_final[mask_test_real]
    )
    r2_rain = r2_score(
        y_reg_test[mask_test_real],
        y_pred_final[mask_test_real]
    )

    # Avaliar também no total (incluindo horas sem chuva)
    rmse_all = np.sqrt(mean_squared_error(y_reg_test, y_pred_final))
    r2_all   = r2_score(y_reg_test, y_pred_final)

    mlflow.log_params({
        "model": "RandomForest", "type": "regressor",
        "strategy": "2stage_clf_then_reg", "version": "v2"
    })
    mlflow.log_metrics({
        "rmse_rain_only": rmse_rain,
        "mae_rain_only":  mae_rain,
        "r2_rain_only":   r2_rain,
        "rmse_all":       rmse_all,
        "r2_all":         r2_all
    })
    mlflow.sklearn.log_model(
        rf_reg, "model",
        input_example=X_train[mask_train].iloc[:5]
    )

    print(f"   RF Regressor v2 (quando chove):")
    print(f"   R²   : {r2_rain:.4f}")
    print(f"   RMSE : {rmse_rain:.4f} mm")
    print(f"   MAE  : {mae_rain:.4f} mm")
    print(f"\n   RF Regressor v2 (todas as horas):")
    print(f"   R²   : {r2_all:.4f}")
    print(f"   RMSE : {rmse_all:.4f} mm")

# COMMAND ----------

# Modelo 2: RandomForest

print("Treinando RandomForest...\n")

# Classificador
with mlflow.start_run(run_name="RandomForest_Classifier") as run:
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        class_weight="balanced",   # trata class imbalance
        random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train, y_clf_train)
    y_pred_rf  = rf_clf.predict(X_test)
    y_prob_rf  = rf_clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_clf_test, y_pred_rf)
    f1   = f1_score(y_clf_test, y_pred_rf)
    auc  = roc_auc_score(y_clf_test, y_prob_rf)
    prec = precision_score(y_clf_test, y_pred_rf)
    rec  = recall_score(y_clf_test, y_pred_rf)

    mlflow.log_params({
        "model": "RandomForest", "type": "classifier",
        "n_estimators": 200, "max_depth": 15,
        "class_weight": "balanced"
    })
    mlflow.log_metrics({
        "accuracy": acc, "f1": f1, "auc_roc": auc,
        "precision": prec, "recall": rec
    })
    mlflow.sklearn.log_model(rf_clf, "model")
    rf_clf_run = run.info.run_id

    print(f"   RandomForest Classifier:")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   F1        : {f1:.4f}")
    print(f"   AUC-ROC   : {auc:.4f}")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")

# Regressor
with mlflow.start_run(run_name="RandomForest_Regressor") as run:
    rf_reg = RandomForestRegressor(
        n_estimators=200, max_depth=15,
        random_state=42, n_jobs=-1
    )
    rf_reg.fit(X_train[mask_train], y_reg_train[mask_train])
    y_pred_rf_reg = np.maximum(rf_reg.predict(X_test), 0)

    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_rf_reg))
    mae  = mean_absolute_error(y_reg_test, y_pred_rf_reg)
    r2   = r2_score(y_reg_test, y_pred_rf_reg)

    mlflow.log_params({
        "model": "RandomForest", "type": "regressor",
        "n_estimators": 200, "trained_on": "rain_only"
    })
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.sklearn.log_model(rf_reg, "model")
    rf_reg_run = run.info.run_id

    print(f"\n   RandomForest Regressor:")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   MAE  : {mae:.4f}")
    print(f"   R²   : {r2:.4f}")

# COMMAND ----------

# Modelo 3: Prophet (séries temporais)

print("Treinando Prophet...\n")

# Prophet usa dados agregados diários
df_prophet = spark.sql("""
    SELECT
        DATE(observation_time)       AS ds,
        SUM(precipitation)           AS y,
        MAX(CASE WHEN precipitation > 0
            THEN 1 ELSE 0 END)       AS teve_chuva
    FROM weather_pipeline.gold.rain_features
    WHERE observation_time < '2024-01-01'
    GROUP BY DATE(observation_time)
    ORDER BY ds
""").toPandas()

df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

with mlflow.start_run(run_name="Prophet_Rain") as run:
    model_prophet = Prophet(
        yearly_seasonality  = True,
        weekly_seasonality  = True,
        daily_seasonality   = False,
        seasonality_mode    = "multiplicative",
        changepoint_prior_scale = 0.1
    )
    model_prophet.fit(df_prophet[["ds","y"]])

    # Prever período de teste (2024–2026)
    future   = model_prophet.make_future_dataframe(
        periods=365*2, freq="D"
    )
    forecast = model_prophet.predict(future)
    forecast_test = forecast[forecast["ds"] >= "2024-01-01"]

    # Métricas diárias
    df_test_daily = spark.sql("""
        SELECT DATE(observation_time) AS ds,
               SUM(precipitation)    AS y_real
        FROM weather_pipeline.gold.rain_features
        WHERE observation_time >= '2024-01-01'
        GROUP BY DATE(observation_time)
        ORDER BY ds
    """).toPandas()
    df_test_daily["ds"] = pd.to_datetime(df_test_daily["ds"])

    merged = df_test_daily.merge(
        forecast_test[["ds","yhat"]], on="ds", how="inner"
    )
    merged["yhat"] = np.maximum(merged["yhat"], 0)

    rmse = np.sqrt(mean_squared_error(merged["y_real"], merged["yhat"]))
    mae  = mean_absolute_error(merged["y_real"], merged["yhat"])
    r2   = r2_score(merged["y_real"], merged["yhat"])

    mlflow.log_params({
        "model": "Prophet",
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True
    })
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    prophet_run = run.info.run_id

    print(f"   Prophet (precipitação diária):")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   MAE  : {mae:.4f}")
    print(f"   R²   : {r2:.4f}")

# COMMAND ----------

# Ranking com métricas corretas
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

clf_runs = []
reg_runs = []
vistos   = set()  # evitar duplicatas

for run in runs:
    params  = run.data.params
    metrics = run.data.metrics
    nome    = run.info.run_name
    versao  = params.get("version", "v1")
    chave   = f"{nome}_{versao}"

    if chave in vistos:
        continue
    vistos.add(chave)

    if params.get("type") == "classifier":
        clf_runs.append({
            "modelo":    nome,
            "f1":        metrics.get("f1", 0),
            "auc_roc":   metrics.get("auc_roc", 0),
            "recall":    metrics.get("recall", 0),
            "precision": metrics.get("precision", 0),
        })
    elif params.get("type") == "regressor":
        # Suporta tanto métricas v1 quanto v2
        r2   = metrics.get("r2_rain_only",  metrics.get("r2",   0))
        rmse = metrics.get("rmse_rain_only", metrics.get("rmse", 0))
        mae  = metrics.get("mae_rain_only",  metrics.get("mae",  0))
        reg_runs.append({
            "modelo": nome,
            "r2":     r2,
            "rmse":   rmse,
            "mae":    mae,
        })
    elif "Prophet" in nome:
        reg_runs.append({
            "modelo": nome,
            "r2":     metrics.get("r2",   0),
            "rmse":   metrics.get("rmse", 0),
            "mae":    metrics.get("mae",  0),
        })

print("=" * 65)
print("  RANKING FINAL — TODOS OS MODELOS")
print("=" * 65)

print("\nCLASSIFICAÇÃO (vai chover na próxima hora?):")
print(f"{'Modelo':<35} {'F1':>8} {'AUC-ROC':>10} {'Recall':>10} {'Precision':>12}")
print("-" * 78)
for r in sorted(clf_runs, key=lambda x: x["f1"], reverse=True):
    print(f"  {r['modelo']:<33} {r['f1']:>8.4f} {r['auc_roc']:>10.4f} "
          f"{r['recall']:>10.4f} {r['precision']:>12.4f}")

print("\nREGRESSÃO (quantos mm vão cair? — avaliado quando chove):")
print(f"{'Modelo':<35} {'R²':>8} {'RMSE(mm)':>10} {'MAE(mm)':>10}")
print("-" * 65)
for r in sorted(reg_runs, key=lambda x: x["r2"], reverse=True):
    print(f"  {r['modelo']:<33} {r['r2']:>8.4f} {r['rmse']:>10.4f} "
          f"{r['mae']:>10.4f}")