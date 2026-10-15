# Databricks notebook source
# MAGIC %md
# MAGIC ### Roda mensalmente — adiciona dados do mês anterior ao treino

# COMMAND ----------

from pyspark.sql import functions as F
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import os

# COMMAND ----------

mlflow.autolog(disable=True)

# Serverless (runtime 19.x): sem URIs explícitos o MLflow tenta ler a config
# spark.mlflow.modelRegistryUri, bloqueada no Spark Connect (CONFIG_NOT_AVAILABLE)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

MLFLOW_TMP = "/Volumes/weather_pipeline/bronze/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP

EXPERIMENT_NAME = "/Users/{}/weather-ml-rain-forecast".format(
    spark.sql("SELECT current_user()").collect()[0][0]
)
mlflow.set_experiment(EXPERIMENT_NAME)

UC_MODEL_NAME = "workspace.default.rain-forecast-birigui"

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

print(" Verificando se retreinamento é necessário...\n")

# Checar quantos novos registros desde último treino
ultimo_treino = spark.sql("""
    SELECT MAX(observation_time) AS ultimo
    FROM weather_pipeline.gold.rain_features
    WHERE observation_time < '2023-01-01'
""").collect()[0][0]

novos = spark.sql(f"""
    SELECT COUNT(*) AS total
    FROM weather_pipeline.silver.weather_clean
    WHERE observation_time > '{ultimo_treino}'
      AND is_forecast = false
""").collect()[0][0]

print(f"   Último treino : {ultimo_treino}")
print(f"   Novos registros: {novos:,}")

# Só retreina se tiver mais de 720 horas novas (1 mês)
if novos < 720:
    print(f"\n Retreinamento não necessário ainda.")
    print(f"   Faltam {720 - novos} registros para 1 mês completo.")
else:
    print(f"\n Retreinando com dados atualizados...\n")

    # Carregar todos os dados até ontem
    from pyspark.sql import functions as F

    df = spark.table("weather_pipeline.gold.rain_features") \
        .filter(F.col("observation_time") < F.date_sub(
            F.current_date(), 1)) \
        .dropna(subset=ML_FEATURES + ["target_will_rain"])

    # Split: últimos 2 anos = teste, resto = treino
    cutoff = F.add_months(F.current_date(), -24)
    train_df = df.filter(F.col("observation_time") < cutoff)
    test_df  = df.filter(F.col("observation_time") >= cutoff)

    X_train = train_df.select(ML_FEATURES).toPandas()
    y_train = train_df.select("target_will_rain") \
                      .toPandas().values.ravel()
    X_test  = test_df.select(ML_FEATURES).toPandas()
    y_test  = test_df.select("target_will_rain") \
                     .toPandas().values.ravel()

    from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                             precision_score, recall_score)
    from sklearn.ensemble import RandomForestClassifier

    with mlflow.start_run(run_name="RF_Retrain_Monthly") as run:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15,
            class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        f1  = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_params({
            "model":        "RandomForestClassifier",
            "type":         "classifier",
            "trigger":      "monthly_retrain",
            "train_size":   len(X_train),
            "test_size":    len(X_test),
        })
        mlflow.log_metrics({"f1": f1, "auc_roc": auc, "accuracy": acc,
                            "precision": prec, "recall": rec})
        # Unity Catalog exige signature para registrar o modelo.
        # input_example segue removido — causa CONFIG_NOT_AVAILABLE no Serverless.
        signature = infer_signature(X_test.head(100), y_pred[:100])
        model_info = mlflow.sklearn.log_model(model, "model", signature=signature)
        run_id = run.info.run_id

    # ------------------------------------------------------------------
    # Model Registry (Unity Catalog): registra a versão e promove o alias
    # "champion" só se o F1 não piorar frente ao champion atual, avaliado
    # no MESMO conjunto de teste (a janela de teste muda a cada mês).
    # ------------------------------------------------------------------
    from mlflow.exceptions import MlflowException
    client = mlflow.tracking.MlflowClient()

    mv = mlflow.register_model(model_info.model_uri, UC_MODEL_NAME)
    client.set_model_version_tag(UC_MODEL_NAME, mv.version, "run_id", run_id)
    client.set_model_version_tag(UC_MODEL_NAME, mv.version, "f1", f"{f1:.4f}")

    try:
        champion = client.get_model_version_by_alias(UC_MODEL_NAME, "champion")
    except MlflowException as e:
        if e.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        champion = None

    if champion is None:
        promote, f1_champion = True, None
        motivo = "ainda não existe champion"
    else:
        champ_model = mlflow.sklearn.load_model(
            f"models:/{UC_MODEL_NAME}@champion")
        f1_champion = f1_score(y_test, champ_model.predict(X_test))
        promote = f1 >= f1_champion
        motivo = (f"F1 novo {f1:.4f} vs champion v{champion.version} "
                  f"{f1_champion:.4f} (mesmo teste)")

    if promote:
        client.set_registered_model_alias(UC_MODEL_NAME, "champion", mv.version)
    print(f"   Versão registrada : {UC_MODEL_NAME} v{mv.version}")
    print(f"   Champion          : {'PROMOVIDA' if promote else 'mantido'} — {motivo}")

    print(f"""
       RETREINAMENTO CONCLUÍDO!
       Accuracy: {acc:.4f}
       F1      : {f1:.4f}
       Precision: {prec:.4f} | Recall: {rec:.4f}
       AUC-ROC : {auc:.4f}
       Versão  : v{mv.version} registrada | champion {'PROMOVIDO' if promote else 'mantido'}
    """)