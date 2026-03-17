# Databricks notebook source
# =============================================================================
# CÉLULA 1 — Imports
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor":  "#161b22",
    "axes.edgecolor":   "#30363d", "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor":  "#ffffff", "xtick.color":     "#8b949e",
    "ytick.color":      "#8b949e", "text.color":      "#c9d1d9",
    "grid.color":       "#21262d", "grid.linestyle":  "--",
    "grid.alpha":       0.5,       "font.family":     "monospace",
})

df_pd = spark.table("weather_pipeline.gold.rain_features").toPandas()
print(f"✅ Feature Store: {len(df_pd):,} registros")

# COMMAND ----------

# =============================================================================
# CÉLULA 2 — Gráfico 1: Distribuição do target + sazonalidade
# =============================================================================
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Target balance
ax1 = fig.add_subplot(gs[0, 0])
counts = df_pd["target_will_rain"].value_counts()
bars   = ax1.bar(["Sem Chuva\n(0)", "Com Chuva\n(1)"],
                 [counts[0], counts[1]],
                 color=["#58a6ff","#3fb950"],
                 edgecolor="#0d1117", linewidth=0.5)
for bar, val in zip(bars, [counts[0], counts[1]]):
    pct = val / len(df_pd) * 100
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1000,
             f"{val:,}\n({pct:.1f}%)",
             ha="center", fontsize=9, color="#ffffff")
ax1.set_title("Distribuição do Target", fontweight="bold")
ax1.set_ylabel("Registros")
ax1.grid(True, axis="y")

# Precipitação por mês
ax2 = fig.add_subplot(gs[0, 1])
rain_month = df_pd.groupby("month")["target_will_rain"].mean() * 100
meses = ["Jan","Fev","Mar","Abr","Mai","Jun",
         "Jul","Ago","Set","Out","Nov","Dez"]
colors_m = ["#3fb950" if v > 20 else "#58a6ff"
            if v > 10 else "#ffa657" for v in rain_month.values]
ax2.bar(meses, rain_month.values, color=colors_m,
        edgecolor="#0d1117", linewidth=0.5)
ax2.axhline(rain_month.mean(), color="#f78166",
            linestyle="--", linewidth=2,
            label=f"Média: {rain_month.mean():.1f}%")
ax2.set_title("% Horas com Chuva por Mês", fontweight="bold")
ax2.set_ylabel("% Horas com Chuva")
ax2.legend(fontsize=8, framealpha=0.2)
ax2.grid(True, axis="y")

# Precipitação por hora do dia
ax3 = fig.add_subplot(gs[0, 2])
rain_hour = df_pd.groupby("hour")["target_will_rain"].mean() * 100
ax3.fill_between(rain_hour.index, rain_hour.values,
                 alpha=0.3, color="#3fb950")
ax3.plot(rain_hour.index, rain_hour.values,
         color="#3fb950", linewidth=2, marker="o", markersize=4)
ax3.set_title("% Horas com Chuva por Hora do Dia", fontweight="bold")
ax3.set_xlabel("Hora (UTC)")
ax3.set_ylabel("% Horas com Chuva")
ax3.set_xticks(range(0, 24, 3))
ax3.grid(True)

# Distribuição da quantidade de chuva (quando chove)
ax4 = fig.add_subplot(gs[1, 0])
rain_only = df_pd[df_pd["target_rain_next_1h"] > 0]["target_rain_next_1h"]
ax4.hist(rain_only, bins=50, color="#3fb950",
         edgecolor="#0d1117", linewidth=0.3, alpha=0.85)
ax4.axvline(rain_only.mean(), color="#ffa657",
            linestyle="--", linewidth=2,
            label=f"Média: {rain_only.mean():.2f}mm")
ax4.axvline(rain_only.median(), color="#f78166",
            linestyle="--", linewidth=2,
            label=f"Mediana: {rain_only.median():.2f}mm")
ax4.set_title("Distribuição da Chuva (quando chove)", fontweight="bold")
ax4.set_xlabel("Precipitação (mm)")
ax4.set_ylabel("Frequência")
ax4.legend(fontsize=8, framealpha=0.2)
ax4.grid(True, axis="y")

# Correlação features vs target
ax5 = fig.add_subplot(gs[1, 1])
num_features = [
    "temperature_2m", "relative_humidity", "windspeed",
    "precip_lag_1h", "precip_lag_3h", "precip_rolling_6h",
    "humidity_lag_1h", "temp_delta_1h", "humidity_delta_1h",
    "is_rainy_season", "is_afternoon"
]
corr = df_pd[num_features + ["target_will_rain"]] \
    .corr()["target_will_rain"] \
    .drop("target_will_rain") \
    .sort_values()
colors_c = ["#f78166" if v < 0 else "#3fb950" for v in corr.values]
ax5.barh(corr.index, corr.values, color=colors_c,
         edgecolor="#0d1117", linewidth=0.4)
ax5.axvline(0, color="#c9d1d9", linewidth=1)
ax5.set_title("Correlação com Target (vai chover?)",
              fontweight="bold")
ax5.set_xlabel("Correlação")
ax5.grid(True, axis="x")

# Heatmap chuva hora x mês
ax6 = fig.add_subplot(gs[1, 2])
pivot = df_pd.pivot_table(
    values="target_will_rain",
    index="hour", columns="month",
    aggfunc="mean"
) * 100
sns.heatmap(pivot, ax=ax6, cmap="YlOrRd",
            annot=False, fmt=".0f",
            linewidths=0.1, linecolor="#0d1117",
            cbar_kws={"label": "% com chuva"})
ax6.set_title("% Chuva por Hora × Mês", fontweight="bold")
ax6.set_xlabel("Mês")
ax6.set_ylabel("Hora")
ax6.set_xticklabels(["J","F","M","A","M","J",
                      "J","A","S","O","N","D"])

plt.suptitle("EDA — Previsão de Chuva Birigui-SP (1940–2026)",
             fontsize=14, fontweight="bold",
             color="#ffffff", y=1.01)
plt.tight_layout()
plt.show()
print("✅ Gráfico EDA gerado")

# COMMAND ----------

# =============================================================================
# CÉLULA 3 — Relatório de qualidade para ML
# =============================================================================
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

print("=" * 60)
print("  RELATÓRIO DE QUALIDADE — FEATURES PARA ML")
print("=" * 60)
print(f"\n{'Feature':<28} {'Nulos':>8} {'Média':>10} "
      f"{'Std':>10} {'Min':>8} {'Max':>8}")
print("-" * 76)
for feat in ML_FEATURES:
    nulos = df_pd[feat].isna().sum()
    media = df_pd[feat].mean()
    std   = df_pd[feat].std()
    mn    = df_pd[feat].min()
    mx    = df_pd[feat].max()
    flag  = " ⚠️" if nulos > 0 else ""
    print(f"{feat:<28} {nulos:>8,} {media:>10.3f} "
          f"{std:>10.3f} {mn:>8.2f} {mx:>8.2f}{flag}")

rain_rate = df_pd["target_will_rain"].mean() * 100
print(f"""
✅ EDA CONCLUÍDO!
   Registros         : {len(df_pd):,}
   Features          : {len(ML_FEATURES)}
   Taxa de chuva     : {rain_rate:.1f}%
   Class imbalance   : {100-rain_rate:.1f}% / {rain_rate:.1f}%
   Próximo passo     : 03_model_training.py
""")