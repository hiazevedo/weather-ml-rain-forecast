# Weather ML Rain Forecast — Birigui-SP

> Modelo de Machine Learning para previsão de chuvas em Birigui-SP com XGBoost, RandomForest e Prophet — treinado com 87 anos de dados históricos

![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-0194E2?style=for-the-badge&logo=databricks&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 📌 Sobre o projeto

Pipeline completo de Machine Learning que utiliza **755.491 registros horários** (1940→2026) para treinar modelos de previsão de chuva para Birigui-SP. Compara XGBoost, RandomForest e Prophet, versiona os modelos no Model Registry do Unity Catalog (alias `champion`) e executa inferência automática a cada hora via Databricks Workflows, alimentando um dashboard em tempo real.

---

## Objetivos do modelo

| Problema | Tipo | Target | Métrica |
|----------|------|--------|---------|
| Vai chover na próxima hora? | Classificação binária | `target_will_rain` | AUC-ROC / F1 |
| Quantos mm vão cair? | Regressão | `target_rain_next_1h` | RMSE / R² |

---

## Arquitetura

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
  → loga o run BestModel_RF_v2_FINAL no MLflow Tracking
           │
           ▼
[05_batch_inference.py]          [06_retrain_model.py]
  Predições próximas 24h           Retreino mensal
  (modelo @champion)               → nova versão no Registry
  → gold.rain_forecast_ml          → promove @champion se F1 ≥ atual
```

---

## Resultados

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

## Feature Engineering — 46 features (variante B)

Definidas num único lugar, o notebook `features.py` (lista `ML_FEATURES` e
função `add_features`), usado via `%run ./features` por 01, 03, 04, 05, 06 e 07.
A inferência usa as colunas que o próprio champion espera (`feature_names_in_`).

| Grupo | Features | Técnica |
|-------|----------|---------|
| **Lag** | temp/precip/humidity lag 1h, 3h, 24h | Janela temporal |
| **Delta** | temp/humidity/precip delta 1h | Variação horária |
| **Rolling** | temp/humidity rolling 6h, precip rolling 3h/6h/24h | Média/soma móvel |
| **Cíclicas** | hour_sin/cos, month_sin/cos | Encoding cíclico |
| **Flags** | is_rainy_season, is_peak_rain_month, is_afternoon | Domínio climatológico |
| **Umidade/estabilidade** | dew_point_2m, dewpoint_depression, tcwv, tcwv_delta_3h, boundary_layer_height | Variáveis extras do ERA5 |
| **Pressão** | pressure_msl, pressure_tend_3h, pressure_tend_24h | Tendência de pressão (frentes) |
| **Nuvens/vento** | cloud_cover (total/low/mid/high), cloud_low_delta_1h, wind_dir_sin/cos | Camadas e direção |

### Experimentos (`gold.ml_experiments`)

| Notebook | Pergunta | Resultado |
|---|---|---|
| `08_experiment_extra_features` | Variáveis extras do ERA5 e o ENSO ajudam? | Extras: F1 0,690 → 0,705, AUC 0,917 → 0,943 (adotado). ENSO: sem ganho na escala horária |
| `09_experiment_mos` | Corrigir a previsão com previsões arquivadas (MOS)? | A 1 dia: API 0,526 · champion 0,517 · MOS 0,528; a 2 dias: MOS 0,548 — aguardando mais dados |
| `10_experiment_fonte_historica` | Treinar com ERA5 puro ou com a série mista (ERA5 → IFS em 2018)? | Empate no que a produção vê (F1 0,7015 vs 0,7042) — mantida a série mista (`best_match`), mesma fonte das horas recentes |
| `11_experiment_inmet_validacao` | Como o Open-Meteo e o champion se saem contra chuva **medida** (INMET) nas cidades das estações? | Open-Meteo marca chuva em 2,4–2,9× mais horas que o pluviômetro (3 de 4 horas "chuvosas" estão secas no INMET), mas acerta o volume mensal (corr. 0,82–0,91). O v9 mantém F1 ≈ 0,69 contra o rótulo do Open-Meteo em outras cidades, mas cai para 0,22–0,46 contra a chuva medida — abaixo da persistência do próprio pluviômetro (0,40–0,61) |
| `12_experiment_regional_inmet` | Treinar com chuva **medida** (INMET) melhora a previsão numa cidade sem estação? | Deixando cada estação de fora (como Birigui): F1 contra chuva medida 0,328 (champion) → **0,427** (regional), AUC 0,849 → 0,894; melhora nas 4 cidades. A persistência do pluviômetro (0,561) segue melhor, mas exige uma estação no local |
| `14_experiment_mos_multimodelo` | Combinar 6 modelos (ECMWF, GFS, ICON, JMA, GEM e best_match), previstos 1 e 2 dias antes, bate a previsão crua da Open-Meteo? | **Sim, nos dois rótulos.** Birigui (rótulo Open-Meteo, últimos 12 meses): F1 0,539 → **0,582** a 1 dia e 0,515 → 0,569 a 2 dias; AUC 0,725 → 0,880. Chuva **medida** (INMET, estação fora do treino; F1 ponderado pelas horas): 0,256 → **0,307** a 1 dia e 0,240 → 0,284 a 2 dias; AUC ~0,67 → ~0,84. Só contar votos dos modelos já ajuda (0,282 a 1 dia). O melhor modelo sozinho costuma ser o ECMWF |
| `15_experiment_merge_validacao` | O MERGE/CPTEC (satélite + pluviômetros) serve de chuva medida para Birigui? | **Não.** Nas estações do INMET bate hora a hora (corr. 0,77, F1 0,81), porque as usa; contra os 5 postos da ANA em Birigui (independentes, 243 dias): corr. diária 0,37 (Open-Meteo 0,57), 45% do volume (Open-Meteo 80%), chuva em 14% dos dias (postos 29%) |

**Top 5 features por correlação com target:**
```
precip_rolling_6h   0.44  ████████████████████████
precip_lag_1h       0.38  ████████████████████
is_rainy_season     0.28  ███████████████
humidity_lag_1h     0.26  ██████████████
precip_lag_3h       0.25  █████████████
```

---

## Class Imbalance

```
SEM CHUVA : 608.546 registros (80.5%)
COM CHUVA : 146.945 registros (19.5%)
```

**Estratégias aplicadas:**
- `class_weight="balanced"` no RandomForest
- Threshold otimizado: **0.65** (vs padrão 0.50)
- Split temporal — não aleatório — para respeitar ordem cronológica

---

## Retreinamento automático

O modelo é retreinado **1x por mês** via Job separado:

```
[weather-ml] Retreinamento Mensal
  Schedule: dia 1 de cada mês às 03h30 (Brasília; no meio da hora, longe do job de 1h)

  update_features  ← atualiza feature store incremental
       │
  retrain_model    ← retreina com histórico completo
                   → registra nova versão no Model Registry (UC)
                   → promove o alias `champion` se o F1 não piorar
```

**Regra de promoção:** o modelo novo é comparado ao `champion` atual **no mesmo
conjunto de teste** (últimos 24 meses — a janela muda a cada mês, então comparar
com o F1 histórico não seria justo). Se `F1_novo >= F1_champion`, o alias passa
para a nova versão; caso contrário a versão fica registrada, mas a inferência
continua usando o champion anterior.

---

## Estrutura do projeto

```
weather-ml-rain-forecast/
├── databricks.yml               # Databricks Asset Bundle — 2 Jobs configurados
├── features.py                  # Definição única das features (variante B)
├── 01_feature_engineering.py    # Feature Store Delta (gold.rain_features)
├── 02_exploratory_analysis.py   # EDA + correlações + heatmap hora×mês
├── 03_model_training.py         # XGBoost + RF + Prophet com MLflow
├── 04_model_evaluation.py       # Matriz confusão + ROC + Registry
├── 05_batch_inference.py        # Predições 24h → gold.rain_forecast_ml
├── 06_retrain_model.py          # Retreino mensal automático
├── 07_update_features.py        # Atualização incremental feature store
├── 08_experiment_extra_features.py  # Experimento: extras do ERA5 + ENSO
├── 09_experiment_mos.py         # Experimento: MOS com previsões arquivadas
├── 10_experiment_fonte_historica.py # Experimento: ERA5 puro vs série mista
├── 11_experiment_inmet_validacao.py # Validação com chuva medida (INMET) — roda no job semanal
├── 12_experiment_regional_inmet.py  # Experimento: modelo regional treinado com chuva medida
├── 13_train_regional_inmet.py       # Treino mensal do modelo regional (chuva medida, INMET)
├── 14_experiment_mos_multimodelo.py # Experimento: MOS com 6 modelos (Previous Runs) contra a API
├── 15_experiment_merge_validacao.py # Experimento: MERGE/CPTEC como chuva medida para Birigui
├── mos.py                           # Features do MOS multimodelo (definição única)
├── 16_train_mos_multimodelo.py      # Treino mensal do MOS multimodelo (versões no Volume)
└── 17_inference_mos_multimodelo.py  # MOS: chance de chuva medida nas próximas 48 h (a cada 3 h)
```

---

## Databricks Jobs

O projeto configura **3 Jobs** via Asset Bundle (`databricks.yml`):

```
[weather-ml] Pipeline Completo (manual)
  feature_engineering → model_training → model_evaluation → retrain_model → batch_inference

[weather-ml] Retreinamento Mensal
  Schedule: dia 1 de cada mês às 03h30
  update_features → retrain_model   ·   train_regional   ·   train_mos

[weather-ml] MOS multimodelo (a cada 3 h)
  Schedule: a cada 3 h, no minuto 20
  inference_mos  ← 6 modelos da Forecast API → gold.rain_forecast_mos
```

Para fazer o deploy:

```bash
databricks bundle deploy
```

---

## Model Registry (Unity Catalog)

Os três modelos ficam registrados no Unity Catalog pelo MLflow (Catalog Explorer →
`workspace` → `default` → Models). Cada um tem o alias `@champion`, que a inferência
carrega, e o limiar de decisão na tag `threshold` da versão.

| Modelo | O que prevê | Champion (nov/2026) | Treino |
|---|---|---|---|
| `workspace.default.rain-forecast-birigui` | Chuva na próxima hora (rótulo Open-Meteo) | v10 — F1 0,718, limiar 0,50 | `06_retrain_model`, mensal |
| `workspace.default.rain-forecast-regional-inmet` | Chuva **medida** na próxima hora | v2 — F1 0,423 (estação fora), limiar 0,70 | `13_train_regional_inmet`, mensal |
| `workspace.default.rain-forecast-mos-multimodelo` | Chuva **medida** nas próximas 48 h, combinando 6 modelos | v2 — F1 0,309 (API 0,259), limiar 0,125 | `16_train_mos_multimodelo`, mensal |

Histórico do `rain-forecast-birigui`:

```
├── v1–v5  → março/2026 (registros manuais, anteriores ao fluxo champion)
├── v6     → RF_Retrain_Monthly  (15/10/2026 — retreino com dados deduplicados)
├── v7     → RF_Retrain_Monthly  (15/10/2026) — 31 features, limiar 0,65, F1 0,687
├── v8     → RF_Retrain_Monthly  (15/10/2026) — não promovido (F1 0,683 < v7)
├── v9     → RF_Retrain_Monthly  (15/10/2026) — variante B, 46 features, limiar 0,55, F1 0,709
└── v10    → RF_Retrain_Monthly  (01/11/2026) — limiar 0,50, F1 0,718 ← @champion
```

- O `06_retrain_model` registra cada retreino e decide a promoção do alias.
- O `05_batch_inference` carrega `models:/workspace.default.rain-forecast-birigui@champion`
  e grava a versão usada na coluna `modelo` de `gold.rain_forecast_ml`.
- **Sem fallback:** se o champion não puder ser carregado, o notebook falha com
  erro explícito (antes, treinava em silêncio um modelo reduzido).
- Em um ambiente novo, rode o `06_retrain_model` **antes** do `05` para criar o
  primeiro champion.

### Modelo regional — chuva medida (INMET)

Modelo: `workspace.default.rain-forecast-regional-inmet` (alias `@champion`), em
paralelo ao champion — não o substitui.

- **Rótulo:** chuva > 0 no pluviômetro das estações automáticas do INMET num raio
  de 80 km (José Bonifácio, Valparaíso, Tupã, Lins) na hora seguinte.
- **Features:** as mesmas 46 do champion, do Open-Meteo no ponto de cada estação.
- **Avaliação:** deixando cada estação de fora (Birigui não tem estação), nos
  últimos 12 meses. v1: F1 0,423 contra chuva medida (champion: 0,345), AUC 0,891,
  limiar 0,70. Métricas em `gold.rain_regional_versions`.
- **Treino:** `13_train_regional_inmet`, mensal (task `train_regional` do job de
  retreino); a nova versão vira champion regional se o F1 não cair mais de 0,02.
- **Produção:** o `05_batch_inference` grava `prob_chuva_medida` /
  `pred_chuva_medida` ao lado da previsão do champion. Se o regional falhar, a
  previsão do champion é gravada e o notebook falha no final (alerta).

### MOS multimodelo — chuva medida combinando 6 modelos

Terceira previsão, ao lado do champion e do regional, para as **próximas 48 h**
(no painel: "Amanhã — chuva medida" e a linha "6 modelos" no gráfico de 48 h).

- **Rótulo:** chuva > 0 no pluviômetro das 4 estações do INMET (como o regional).
- **Features** (`mos.py`, 35): a previsão de cada um dos 6 modelos (ECMWF IFS, GFS,
  ICON, JMA, GEM e best_match) para a hora e as vizinhas, nuvens, umidade, CAPE e o
  consenso entre eles, mais hora e mês. No treino vêm da rodada de 1 dia antes
  (Previous Runs API); em produção, da previsão atual (Forecast API).
- **Avaliação:** cada estação fora do treino, nos últimos 12 meses com rótulo, contra
  a previsão crua da API nas mesmas horas. v2 (08/11/2026): F1 **0,309** contra 0,259
  da API, AUC 0,837, limiar 12,5% (chuva medida em 6,6% das horas do teste).
- **Treino:** `16_train_mos_multimodelo`, mensal (task `train_mos`). Uma versão só é
  promovida se bater a API e o F1 não cair mais de 0,02 em relação à promovida atual.
- **Produção:** `17_inference_mos_multimodelo`, a cada 3 h → `gold.rain_forecast_mos`
  (última previsão) e `gold.rain_forecast_mos_history` (todas, para medir o acerto).
  A resposta crua da Open-Meteo fica no Volume `bronze.forecast_multimodelo`.
- **Registro:** no Unity Catalog, `workspace.default.rain-forecast-mos-multimodelo`
  (alias `@champion`, limiar na tag `threshold`); métricas por versão também em
  `gold.rain_mos_versions` (painel). A v1 foi treinada numa execução avulsa e não carrega no ambiente do job
  (ver as notas abaixo); a v2 é a primeira treinada pelo próprio job.
- **Limite:** o modelo aprendeu com a rodada de 1 dia antes; nas primeiras horas a
  antecedência é menor e o acerto medido pode não valer igual.

### Notas do Serverless (MLflow 3.x)

- **Execução avulsa ≠ job:** uma execução avulsa (`jobs submit`, sem ambiente
  definido) roda num ambiente serverless mais antigo, com MLflow 2.11, que não
  consegue registrar modelos no Unity Catalog (upload negado, S3 AccessDenied) e
  tem outra versão do scikit-learn. Um modelo treinado ali não carrega no job
  (`Can't get attribute '__pyx_unpickle_…'`). Treine sempre pelo job — para rodar
  só uma task: `databricks jobs run-now --json '{"job_id": …, "only": ["train_mos"]}'`.
  O 16 e o 17 instalam `mlflow>=3.1` para não depender disso no registro.
- É necessário `mlflow.set_tracking_uri("databricks")` e
  `mlflow.set_registry_uri("databricks-uc")` antes de qualquer chamada; sem isso
  o MLflow lê `spark.mlflow.modelRegistryUri`, bloqueada no Spark Connect
  (`CONFIG_NOT_AVAILABLE`).
- O Unity Catalog exige **signature** no modelo (`infer_signature`);
  `input_example` segue evitado (mesmo erro de configuração).
- No MLflow 3 o `log_model` cria um *LoggedModel* (`models:/m-…`), e não o
  artefato `runs:/<run_id>/model` do MLflow 2 — não use `runs:/` para carregar.

---

## Integração com Workflow principal

O notebook `05_batch_inference.py` roda automaticamente dentro do pipeline de previsão:

```
[weather] Pipeline Completo (1h)
  Schedule: a cada hora cheia (Brasília)

  coleta_previsao  ← coleta API Open-Meteo
       │
  dlt_pipeline     ← executa DLT Bronze→Silver→Gold
       │
  gold_today       ← atualiza weather_today e rain_alert
       │
  update_features  ← atualiza feature store
       │
  ml_inference     ← predições RandomForest 24h
```

---

## Stack técnica

| Tecnologia | Uso |
|------------|-----|
| **Databricks Free Edition** | Ambiente Serverless AWS |
| **Unity Catalog** | Feature Store + Model Registry storage |
| **MLflow** | Experiment tracking + Model Registry |
| **Scikit-learn** | RandomForest + GradientBoosting |
| **Prophet** | Séries temporais sazonais |
| **Spark ML** | Feature engineering distribuído |
| **Delta Lake** | Feature Store + Predictions table |
| **Databricks Asset Bundles** | 2 Jobs como código |
| **Databricks Workflows** | Inferência de hora em hora + retreino mensal |

---

## Como reproduzir

### Pré-requisitos
- Projeto `weather-dlt-pipeline` executado
- Tabela `weather_pipeline.silver.weather_clean` populada
- Databricks CLI instalado e configurado

### Passo a passo

```bash
# 1. Clone o repositório
git clone https://github.com/hiazevedo/weather-ml-rain-forecast.git
cd weather-ml-rain-forecast

# 2. Deploy via Asset Bundle
databricks bundle deploy
```

Ou execute os notebooks manualmente na ordem:

```
01_feature_engineering.py    # Feature engineering (~10 min — 755k registros)
02_exploratory_analysis.py   # EDA (opcional)
03_model_training.py         # Treinar modelos (~30 min)
04_model_evaluation.py       # Avaliar e logar o run final no MLflow
06_retrain_model.py          # Registrar versão e promover @champion (antes do 05)
05_batch_inference.py        # Rodar inferência com o @champion
```

### Unity Catalog

```
Catalog : weather_pipeline
Schemas : silver | gold
Feature Store : gold.rain_features
Predictions   : gold.rain_forecast_ml
```

---

## Decisões técnicas

**Por que split temporal e não aleatório?**
Dados meteorológicos têm dependência temporal forte — horas consecutivas são altamente correlacionadas. Split aleatório causaria data leakage, inflando artificialmente as métricas. O split temporal (treino 1940–2022, teste 2024–2026) simula o uso real do modelo.

**Por que threshold 0.65 e não 0.50?**
A análise de threshold mostrou que 0.65 maximiza o F1 score, equilibrando precision e recall. Com threshold 0.50, o modelo gera muitos falsos alarmes. Com 0.65, os alertas de chuva são mais confiáveis.

**Por que o regressor tem R² baixo?**
Precipitação horária tem distribuição extremamente assimétrica — 80% dos valores são zero e eventos extremos são raros. O classificador (AUC 0.93) resolve o problema prático de "vai chover?" com alta confiança.

---

## Portfólio

Este projeto faz parte do [Databricks Data Engineering Portfolio](https://github.com/hiazevedo/databricks-portfolio), uma série de projetos práticos cobrindo o ciclo completo de engenharia de dados com Databricks.

| # | Projeto | Tema |
|---|---------|------|
| 1 | [fuel-price-pipeline-br](https://github.com/hiazevedo/fuel-price-pipeline-br) | Batch · Medallion · ANP |
| 2 | [earthquake-streaming-pipeline](https://github.com/hiazevedo/earthquake-streaming-pipeline) | Streaming · Auto Loader · USGS |
| 3 | [earthquake-ml-pipeline](https://github.com/hiazevedo/earthquake-ml-pipeline) | ML · MLflow · Spark ML |
| 4 | [weather-dlt-pipeline](https://github.com/hiazevedo/weather-dlt-pipeline) | DLT · Workflows · Open-Meteo |
| 5 | **weather-ml-rain-forecast** ← você está aqui | ML Avançado · Previsão de Chuva |
