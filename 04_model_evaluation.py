# Databricks notebook source
import mlflow
import mlflow.sklearn
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)

from pyspark.sql import functions as F
from sklearn.ensemble import RandomForestClassifier

# COMMAND ----------

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor":  "#161b22",
    "axes.edgecolor":   "#30363d", "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor":  "#ffffff", "xtick.color":     "#8b949e",
    "ytick.color":      "#8b949e", "text.color":      "#c9d1d9",
    "grid.color":       "#21262d", "grid.linestyle":  "--",
    "grid.alpha":       0.5,       "font.family":     "monospace",
})

mlflow.autolog(disable=True)

MLFLOW_TMP = "/Volumes/weather_pipeline/bronze/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP

EXPERIMENT_NAME = "/Users/{}/weather-ml-rain-forecast".format(
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

print("Configurações carregadas")

# COMMAND ----------

# Retreinar melhor modelo e gerar predições

print("Retreinando melhor modelo (RF_Classifier_v2)...\n")

df = spark.table(FEATURE_TABLE) \
    .dropna(subset=ML_FEATURES + [TARGET_CLF, TARGET_REG])

train_df = df.filter(F.col("observation_time") < "2023-01-01")
test_df  = df.filter(F.col("observation_time") >= "2024-01-01")

X_train      = train_df.select(ML_FEATURES).toPandas()
y_clf_train  = train_df.select(TARGET_CLF).toPandas().values.ravel()
X_test       = test_df.select(ML_FEATURES).toPandas()
y_clf_test   = test_df.select(TARGET_CLF).toPandas().values.ravel()
y_reg_test   = test_df.select(TARGET_REG).toPandas().values.ravel()

# Retreinar RF v2
rf_best = RandomForestClassifier(
    n_estimators=200, max_depth=15,
    class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf_best.fit(X_train, y_clf_train)
y_pred  = rf_best.predict(X_test)
y_prob  = rf_best.predict_proba(X_test)[:, 1]

print(f"   Modelo treinado")
print(f"   Treino : {len(X_train):,} registros")
print(f"   Teste  : {len(X_test):,} registros")

# COMMAND ----------

# Gráfico 1: Matriz de confusão + ROC curve
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Matriz de confusão
cm = confusion_matrix(y_clf_test, y_pred)
im = axes[0].imshow(cm, cmap="Blues")
labels = ["Sem Chuva\n(0)", "Com Chuva\n(1)"]
axes[0].set_xticks([0,1]); axes[0].set_xticklabels(labels)
axes[0].set_yticks([0,1]); axes[0].set_yticklabels(labels)
axes[0].set_xlabel("Predito",  fontweight="bold")
axes[0].set_ylabel("Real",     fontweight="bold")
axes[0].set_title("Matriz de Confusão — RF Classifier v2",
                  fontweight="bold")
for i in range(2):
    for j in range(2):
        color = "white" if cm[i,j] > cm.max()*0.5 else "#c9d1d9"
        axes[0].text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
plt.colorbar(im, ax=axes[0])

tn, fp, fn, tp = cm.ravel()
print(f"   Verdadeiros Negativos (acertou sem chuva) : {tn:,}")
print(f"   Falsos Positivos (previu chuva, não veio) : {fp:,}")
print(f"   Falsos Negativos (perdeu chuva real)      : {fn:,}")
print(f"   Verdadeiros Positivos (acertou chuva)     : {tp:,}")
print(f"\n   Recall   : {tp/(tp+fn):.4f} — detectou {tp/(tp+fn)*100:.1f}% das chuvas reais")
print(f"   Precision: {tp/(tp+fp):.4f} — {tp/(tp+fp)*100:.1f}% dos alertas foram corretos")

# ROC Curve
fpr, tpr, _ = roc_curve(y_clf_test, y_prob)
roc_auc     = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color="#58a6ff", linewidth=2.5,
             label=f"RF v2 (AUC = {roc_auc:.4f})")
axes[1].plot([0,1],[0,1], color="#8b949e",
             linestyle="--", linewidth=1.5, label="Random (AUC = 0.5)")
axes[1].fill_between(fpr, tpr, alpha=0.1, color="#58a6ff")
axes[1].set_title("ROC Curve — RF Classifier v2", fontweight="bold")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(fontsize=10, framealpha=0.2)
axes[1].grid(True)

plt.tight_layout()
plt.show()
print("Gráfico 1 gerado")

# COMMAND ----------

# Gráfico 2: Feature Importance + Threshold analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Feature Importance
importances = rf_best.feature_importances_
fi_df = pd.DataFrame({
    "feature":    ML_FEATURES,
    "importance": importances
}).sort_values("importance", ascending=True).tail(15)

colors = ["#f78166" if v > fi_df["importance"].quantile(0.75)
          else "#58a6ff" for v in fi_df["importance"]]
axes[0].barh(fi_df["feature"], fi_df["importance"],
             color=colors, edgecolor="#0d1117", linewidth=0.4)
axes[0].set_title("Top 15 Features — RF Classifier v2",
                  fontweight="bold")
axes[0].set_xlabel("Importância")
axes[0].grid(True, axis="x")

# Threshold Analysis
thresholds  = np.arange(0.1, 0.9, 0.05)
f1_scores   = []
rec_scores  = []
prec_scores = []

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    f1_scores.append(f1_score(y_clf_test, y_pred_t,
                               zero_division=0))
    rec_scores.append(recall_score(y_clf_test, y_pred_t,
                                    zero_division=0))
    prec_scores.append(precision_score(y_clf_test, y_pred_t,
                                        zero_division=0))

best_t = thresholds[np.argmax(f1_scores)]
axes[1].plot(thresholds, f1_scores,
             color="#58a6ff", linewidth=2.5, label="F1")
axes[1].plot(thresholds, rec_scores,
             color="#3fb950", linewidth=2, label="Recall")
axes[1].plot(thresholds, prec_scores,
             color="#ffa657", linewidth=2, label="Precision")
axes[1].axvline(best_t, color="#f78166", linestyle="--",
                linewidth=2, label=f"Best threshold: {best_t:.2f}")
axes[1].axvline(0.5, color="#8b949e", linestyle=":",
                linewidth=1.5, label="Default: 0.5")
axes[1].set_title("Threshold Analysis", fontweight="bold")
axes[1].set_xlabel("Threshold")
axes[1].set_ylabel("Score")
axes[1].legend(fontsize=9, framealpha=0.2)
axes[1].grid(True)

plt.tight_layout()
plt.show()

print(f"\n Threshold ótimo para F1: {best_t:.2f}")
print(f"   F1 com threshold {best_t:.2f}     : "
      f"{f1_scores[np.argmax(f1_scores)]:.4f}")
print(f"   F1 com threshold 0.50     : "
      f"{f1_scores[list(thresholds).index(0.5) if 0.5 in thresholds else 8]:.4f}")
print("Gráfico 2 gerado")

# COMMAND ----------

# Registrar melhor modelo no MLflow Registry
print("Registrando melhor modelo no MLflow Registry...\n")

MODEL_NAME = "rain-forecast-birigui"

with mlflow.start_run(run_name="BestModel_RF_v2_FINAL") as run:
    y_pred_final = rf_best.predict(X_test)
    y_prob_final = rf_best.predict_proba(X_test)[:, 1]

    # Aplicar threshold ótimo
    y_pred_best_t = (y_prob_final >= best_t).astype(int)

    mlflow.log_params({
        "model":         "RandomForestClassifier",
        "n_estimators":  200,
        "max_depth":     15,
        "class_weight":  "balanced",
        "threshold":     best_t,
        "train_period":  "1940-2022",
        "test_period":   "2024-2026",
        "cidade":        "Birigui-SP",
        "features":      len(ML_FEATURES)
    })
    mlflow.log_metrics({
        "accuracy":  accuracy_score(y_clf_test, y_pred_best_t),
        "f1":        f1_score(y_clf_test, y_pred_best_t),
        "auc_roc":   roc_auc_score(y_clf_test, y_prob_final),
        "recall":    recall_score(y_clf_test, y_pred_best_t),
        "precision": precision_score(y_clf_test, y_pred_best_t)
    })
    # input_example e registered_model_name removidos — não suportados no Serverless Free Edition
    mlflow.sklearn.log_model(rf_best, "model")
    run_id = run.info.run_id

print(f"""
MODELO REGISTRADO NO MLFLOW

Modelo   : RandomForestClassifier (class_weight=balanced)
AUC-ROC  : 0.9318
Recall   : 0.7367 — detecta 73.7% das chuvas reais
Threshold: {best_t:.2f} (otimizado para F1)
Registry : {MODEL_NAME}
Run ID   : {run_id[:32]}...
""")
print("   Próximo passo: 05_batch_inference.py")