# Databricks notebook source
# MAGIC %run ./features

# COMMAND ----------

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

# ML_FEATURES vem de ./features (variante B: base + extras)

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

    # Split temporal: os últimos 24 meses ficam fora do treino.
    #   - calibração (meses -24 a -12): escolhe o limiar de decisão (máx. F1)
    #   - teste      (últimos 12 meses): mede o desempenho NESSE limiar
    # Escolher e medir o limiar no mesmo dado daria métricas otimistas.
    cutoff  = F.add_months(F.current_date(), -24)
    cal_cut = F.add_months(F.current_date(), -12)
    train_df = df.filter(F.col("observation_time") < cutoff)
    cal_df   = df.filter((F.col("observation_time") >= cutoff) &
                         (F.col("observation_time") <  cal_cut))
    test_df  = df.filter(F.col("observation_time") >= cal_cut)

    X_train = train_df.select(ML_FEATURES).toPandas()
    y_train = train_df.select("target_will_rain") \
                      .toPandas().values.ravel()
    X_cal   = cal_df.select(ML_FEATURES).toPandas()
    y_cal   = cal_df.select("target_will_rain") \
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

        # Limiar de decisão: o de maior F1 na janela de calibração.
        # É gravado na versão do modelo e usado pelo 05_batch_inference.
        import numpy as np
        p_cal = model.predict_proba(X_cal)[:, 1]
        candidatos = np.round(np.arange(0.30, 0.91, 0.05), 2)
        threshold = float(max(
            candidatos, key=lambda t: f1_score(y_cal, (p_cal >= t).astype(int))))

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

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
            "cal_size":     len(X_cal),
            "test_size":    len(X_test),
            "threshold":    threshold,
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
    client.set_model_version_tag(UC_MODEL_NAME, mv.version, "threshold", f"{threshold:.2f}")

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
        # Cada modelo é avaliado no próprio limiar (o que usaria em produção)
        champ_model = mlflow.sklearn.load_model(
            f"models:/{UC_MODEL_NAME}@champion")
        champ_thr = float(champion.tags["threshold"])
        # o champion pode usar outro conjunto de features (ex.: v7 = 31 base)
        champ_feats = list(champ_model.feature_names_in_)
        X_test_champ = test_df.select(champ_feats).toPandas()
        f1_champion = f1_score(
            y_test, (champ_model.predict_proba(X_test_champ)[:, 1] >= champ_thr).astype(int))
        promote = f1 >= f1_champion
        motivo = (f"F1 novo {f1:.4f} (limiar {threshold:.2f}) vs champion "
                  f"v{champion.version} {f1_champion:.4f} (limiar {champ_thr:.2f}), mesmo teste")

    if promote:
        client.set_registered_model_alias(UC_MODEL_NAME, "champion", mv.version)

    # Métricas de teste por versão, no limiar da versão (consumidas pelo dashboard)
    from datetime import datetime
    VERSIONS_TABLE = "weather_pipeline.gold.rain_model_versions"
    if "threshold" not in spark.table(VERSIONS_TABLE).columns:
        spark.sql(f"ALTER TABLE {VERSIONS_TABLE} ADD COLUMNS (threshold DOUBLE)")
    spark.createDataFrame([(
        int(mv.version), run_id, datetime.now(), bool(promote),
        float(f1), float(acc), float(prec), float(rec), float(auc),
        int(len(X_train)), int(len(X_test)), float(y_test.mean()), threshold,
    )], "version int, run_id string, created_at timestamp, promoted boolean, "
        "f1 double, accuracy double, `precision` double, recall double, auc_roc double, "
        "train_size int, test_size int, test_rain_rate double, threshold double"
    ).write.mode("append").saveAsTable(VERSIONS_TABLE)

    # Champion mantido: atualiza as métricas dele com a avaliação de hoje
    # (mesmo teste, limiar dele), para o dashboard comparar com a produção.
    if not promote:
        p_ch = champ_model.predict_proba(X_test_champ)[:, 1]
        y_ch = (p_ch >= champ_thr).astype(int)
        spark.createDataFrame([(
            int(champion.version), float(f1_champion),
            float(accuracy_score(y_test, y_ch)), float(precision_score(y_test, y_ch)),
            float(recall_score(y_test, y_ch)), float(roc_auc_score(y_test, p_ch)),
            int(len(X_test)), float(y_test.mean()), champ_thr,
        )], "version int, f1 double, accuracy double, `precision` double, recall double, "
            "auc_roc double, test_size int, test_rain_rate double, threshold double"
        ).createOrReplaceTempView("_champion_eval")
        spark.sql(f"""
            MERGE INTO {VERSIONS_TABLE} t
            USING _champion_eval s ON t.version = s.version
            WHEN MATCHED THEN UPDATE SET
                t.f1 = s.f1, t.accuracy = s.accuracy, t.`precision` = s.`precision`,
                t.recall = s.recall, t.auc_roc = s.auc_roc, t.test_size = s.test_size,
                t.test_rain_rate = s.test_rain_rate, t.threshold = s.threshold
        """)

    print(f"   Versão registrada : {UC_MODEL_NAME} v{mv.version}")
    print(f"   Champion          : {'PROMOVIDA' if promote else 'mantido'} — {motivo}")

    print(f"""
       RETREINAMENTO CONCLUÍDO!
       Accuracy: {acc:.4f}
       F1      : {f1:.4f}
       Precision: {prec:.4f} | Recall: {rec:.4f}
       AUC-ROC : {auc:.4f}
       Limiar  : {threshold:.2f} (máx. F1 na calibração; métricas acima no teste de 12 meses)
       Versão  : v{mv.version} registrada | champion {'PROMOVIDO' if promote else 'mantido'}
    """)