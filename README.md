# 🤖 weather-ml-rain-forecast

> Modelo de Machine Learning para previsão de chuvas em Birigui-SP com XGBoost, RandomForest e Prophet — treinado com 87 anos de dados históricos

![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 📌 Sobre o projeto

Pipeline completo de Machine Learning que utiliza **755.491 registros horários** (1940→2026) para treinar modelos de previsão de chuva para Birigui-SP. Compara XGBoost, RandomForest e Prophet, registra o melhor modelo no MLflow Registry e executa inferência automática a cada 6 horas via Databricks Workflows, alimentando um dashboard em tempo real.

---

## 🎯 Objetivos do modelo

| Problema | Tipo | Target | Métrica |
|----------|------|--------|---------|
| Vai chover na próxima hora? | Classificação binária | `target_will_rain` | AUC-ROC / F1 |
| Quantos mm vão cair? | Regressão | `target_rain_next_1h` | RMSE / R² |

---

## 🏗️ Arquitetura

```
[weather_pipeline.silver.weather_clean]
  87 anos × 755k registros horários
           │
           ▼
[01_feature_engineering.py]
  31 features: lag, rolling, delta, cíclicas, flags
  → gold.rain_features
           │
           ▼
[02_exploratory_analysis.py]
  EDA: sazonalidade, correlações, heatmap hora×mês
           │
           ▼
[03_model_training.py]
  ┌──────────────┬──────────────┬──────────┐
  │   XGBoost    │ RandomForest │  Prophet │
  │  CLF + REG   │  CLF + REG   │   REG    │
  └──────┬───────┴──────┬───────┴────┬─────┘
         └──────────────┼────────────┘
                        ▼
              MLflow Tracking (5 runs)
           │
           ▼
[04_model_evaluation.py]
  Matriz confusão + ROC + Feature Importance
  → MLflow Registry: rain-forecast-birigui v1
           │
           ▼
[05_batch_inference.py]          [06_retrain_model.py]
  Predições próximas 24h           Retreino mensal
  → gold.rain_forecast_ml          → Registry v2, v3...
```

---

## 📊 Resultados

### Classificação — vai chover na próxima hora?

| Modelo | F1 | AUC-ROC | Recall | Precision |
|--------|----|---------|----|-----------|
| 🥇 **RandomForest v2** | 0.6816 | **0.9318** | **0.7367** | 0.6342 |
| XGBoost | 0.6831 | 0.9320 | 0.6494 | 0.7205 |

**Por que RandomForest foi escolhido?**
Apesar do XGBoost ter F1 ligeiramente maior, o RandomForest v2 tem **recall superior (0.7367 vs 0.6494)** — detecta mais chuvas reais. Para previsão meteorológica, é mais importante avisar chuva que não veio do que não avisar chuva que vem.

### Regressão — quantos mm vão cair?

| Modelo | R² | RMSE |
|--------|----|------|
| 🥇 Prophet | 0.1078 | 6.16mm |

> Precipitação é um dos fenômenos mais difíceis de prever em quantidade exata — variabilidade natural alta e eventos extremos raros. O classificador (AUC 0.93) é o produto principal deste projeto.

---

## 🔧 Feature Engineering — 31 features

| Grupo | Features | Técnica |
|-------|----------|---------|
| **Lag** | temp/precip/humidity lag 1h, 3h, 24h | Janela temporal |
| **Delta** | temp/humidity/precip delta 1h | Variação horária |
| **Rolling** | temp/humidity rolling 6h, precip rolling 3h/6h/24h | Média/soma móvel |
| **Cíclicas** | hour_sin/cos, month_sin/cos | Encoding cíclico |
| **Flags** | is_rainy_season, is_peak_rain_month, is_afternoon | Domínio climatológico |

**Top 5 features por correlação com target:**
```
precip_rolling_6h   0.44  ████████████████████████
precip_lag_1h       0.38  ████████████████████
is_rainy_season     0.28  ███████████████
humidity_lag_1h     0.26  ██████████████
precip_lag_3h       0.25  █████████████
```

---

## ⚠️ Class Imbalance

```
SEM CHUVA : 608.546 registros (80.5%)
COM CHUVA : 146.945 registros (19.5%)
```

**Estratégias aplicadas:**
- `class_weight="balanced"` no RandomForest
- Threshold otimizado: **0.65** (vs padrão 0.50)
- Split temporal — não aleatório — para respeitar ordem cronológica

---

## 🔄 Retreinamento automático

O modelo é retreinado **1x por mês** via Job separado:

```
weather-model-retrain
  Schedule: dia 1 de cada mês às 02h (Brasília)
  Task: 06_retrain_model.py

Lógica de retreinamento:
  ├── Verifica se há +720h de dados novos (1 mês)
  ├── Se sim → retreina com histórico completo atualizado
  └── Registra nova versão no MLflow Registry
```

---

## 🗂️ Estrutura do projeto

```
weather-ml-rain-forecast/
├── 01_feature_engineering.py   # 31 features + Feature Store Delta
├── 02_exploratory_analysis.py  # EDA + correlações + heatmap hora×mês
├── 03_model_training.py        # XGBoost + RF + Prophet com MLflow
├── 04_model_evaluation.py      # Matriz confusão + ROC + Registry
├── 05_batch_inference.py       # Predições 24h → gold.rain_forecast_ml
├── 06_retrain_model.py         # Retreino mensal automático
└── 07_update_features.py       # Atualização incremental feature store
```

---

## 📦 MLflow Registry

```
rain-forecast-birigui
├── v1  → BestModel_RF_v2_FINAL   (treinado manualmente)
└── v2  → RF_Retrain_Monthly      (primeiro retreino mensal)
```

---

## ⏰ Integração com Workflow

O notebook `05_batch_inference.py` roda automaticamente dentro do pipeline principal:

```
weather-pipeline-scheduler (4x/dia)
  ├── 01_collect_forecast
  ├── 02_run_dlt_pipeline
  ├── 03_build_gold_today
  ├── 07_update_features    ← atualiza feature store
  └── 05_batch_inference    ← predições ML atualizadas
```

---

## 🛠️ Stack técnica

| Tecnologia | Uso |
|------------|-----|
| **Databricks Free Edition** | Ambiente Serverless AWS |
| **MLflow** | Experiment tracking + Model Registry |
| **Scikit-learn** | RandomForest + GradientBoosting |
| **Prophet** | Séries temporais sazonais |
| **Spark ML** | Feature engineering distribuído |
| **Delta Lake** | Feature Store + Predictions table |
| **Databricks Workflows** | Inferência 4x/dia + retreino mensal |

---

## 🚀 Como reproduzir

### Pré-requisitos
- Projeto `weather-dlt-pipeline-birigui` executado
- Tabela `weather_pipeline.silver.weather_clean` populada

### Passo a passo

```bash
# 1. Feature engineering (~10 min — 755k registros)
01_feature_engineering.py

# 2. EDA (opcional)
02_exploratory_analysis.py

# 3. Treinar modelos (~30 min)
03_model_training.py

# 4. Avaliar e registrar no MLflow
04_model_evaluation.py

# 5. Rodar inferência
05_batch_inference.py

# 6. Configurar retreino mensal
06_retrain_model.py
```

### Configuração MLflow para Serverless

```python
import os
os.environ["MLFLOW_DFS_TMP"] = "/Volumes/weather_pipeline/bronze/mlflow_tmp"

EXPERIMENT_NAME = "/Users/{}/weather-ml-rain-forecast/weather-ml-rain-forecast".format(
    spark.sql("SELECT current_user()").collect()[0][0]
)
mlflow.set_experiment(EXPERIMENT_NAME)
```

---

## ⚙️ Decisões técnicas

**Por que split temporal e não aleatório?**
Dados meteorológicos têm dependência temporal forte — horas consecutivas são altamente correlacionadas. Split aleatório causaria data leakage, inflando artificialmente as métricas. O split temporal (treino 1940–2022, teste 2024–2026) simula o uso real do modelo.

**Por que threshold 0.65 e não 0.50?**
A análise de threshold mostrou que 0.65 maximiza o F1 score, equilibrando precision e recall. Com threshold 0.50, o modelo gera muitos falsos alarmes. Com 0.65, os alertas de chuva são mais confiáveis.

**Por que o regressor tem R² baixo?**
Precipitação horária tem distribuição extremamente assimétrica — 80% dos valores são zero e eventos extremos são raros. O classificador (AUC 0.93) resolve o problema prático de "vai chover?" com alta confiança.

---

## 🔗 Projetos relacionados

| # | Projeto | Skills |
|---|---------|--------|
| 1 | [fuel-price-pipeline-br](https://github.com/hiazevedo/fuel-price-pipeline-br) | Batch, Medallion |
| 2 | [earthquake-streaming-pipeline](https://github.com/hiazevedo/earthquake-streaming-pipeline) | Streaming, Auto Loader |
| 3 | [earthquake-ml-pipeline](https://github.com/hiazevedo/earthquake-ml-pipeline) | MLflow, RandomForest |
| 4 | [weather-dlt-pipeline](https://github.com/hiazevedo/weather-dlt-pipeline) | Delta Live Tables |
| 5 | **weather-ml-rain-forecast** ← você está aqui | ML Séries Temporais |

---
