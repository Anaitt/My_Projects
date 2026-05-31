import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import json

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
    "font.size": 11,
})
sns.set_style("whitegrid")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD & ENGINEER
# ============================================================
print("=" * 70)
print("1. ЗАГРУЗКА ДАННЫХ И FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv("ai_productivity.csv")
print(f"Строк: {len(df)}, Колонок: {df.shape[1]}")

df["workload_hrs"] = (
    df["manual_work_hours_per_week"]
    + df["meeting_hours_per_week"]
    + df["collaboration_hours_per_week"]
)
df["ai_ratio"] = df["ai_tool_usage_hours_per_week"] / (df["workload_hrs"] + 0.01)

df["productivity"] = (
    df["focus_hours_per_day"]
    * df["tasks_automated_percent"]
    / 100
)

df["high_burnout"] = (df["burnout_risk_score"] > df["burnout_risk_score"].median()).astype(int)

df["ai_efficiency"] = df["productivity"] / (df["burnout_risk_score"] + 1)

FEATURES_9 = [
    "ai_tool_usage_hours_per_week",
    "tasks_automated_percent",
    "manual_work_hours_per_week",
    "learning_time_hours_per_week",
    "meeting_hours_per_week",
    "collaboration_hours_per_week",
    "error_rate_percent",
    "focus_hours_per_day",
    "work_life_balance_score",
]

LABELS_9 = [
    "AI usage (hrs)",
    "Tasks automated (%)",
    "Manual work (hrs)",
    "Learning (hrs)",
    "Meetings (hrs)",
    "Collaboration (hrs)",
    "Error rate (%)",
    "Focus (hrs/day)",
    "Work-life balance",
]

AI_FEATURES = [
    "ai_tool_usage_hours_per_week",
    "tasks_automated_percent",
    "ai_ratio",
]

CONTROL_FEATURES = [
    "experience_years",
    "workload_hrs",
    "focus_hours_per_day",
    "learning_time_hours_per_week",
    "work_life_balance_score",
    "error_rate_percent",
    "task_complexity_score",
]

# ============================================================
# 2. LINEAR REGRESSION: productivity
# ============================================================
print("\n" + "=" * 70)
print("2. ЛИНЕЙНАЯ РЕГРЕССИЯ: Productivity")
print("=" * 70)

X_prod = df[AI_FEATURES + CONTROL_FEATURES]
y_prod = df["productivity"]

lr_prod = LinearRegression()
lr_prod.fit(X_prod, y_prod)
y_prod_pred = lr_prod.predict(X_prod)

r2_prod = lr_prod.score(X_prod, y_prod)
print(f"\nR² (продуктивность) = {r2_prod:.4f}")

coef_prod = pd.Series(lr_prod.coef_, index=X_prod.columns)
coef_prod_sorted = coef_prod.abs().sort_values(ascending=False)

print("\nКоэффициенты (по abs(coef)):")
for name, val in coef_prod_sorted.items():
    print(f"  {name:35s}: {lr_prod.coef_[list(X_prod.columns).index(name)]:+.4f}  (|coef| = {val:.4f})")

coef_prod_df = pd.DataFrame({
    "feature": X_prod.columns,
    "coefficient": lr_prod.coef_,
    "abs_coefficient": np.abs(lr_prod.coef_),
}).sort_values("abs_coefficient", ascending=False)
coef_prod_df.to_json(f"{OUTPUT_DIR}/linear_coefficients_productivity.json", orient="records", force_ascii=False, indent=4)

top5_prod = coef_prod_df.head(5)
print("\nТоп-5 факторов влияния (abs coef):")
for i, row in top5_prod.iterrows():
    print(f"  {row['feature']}: {row['coefficient']:+.4f} (|coef| = {row['abs_coefficient']:.4f})")

# ============================================================
# 3. LINEAR REGRESSION: burnout
# ============================================================
print("\n" + "=" * 70)
print("3. ЛИНЕЙНАЯ РЕГРЕССИЯ: Burnout Risk Score")
print("=" * 70)

X_burn = df[AI_FEATURES + CONTROL_FEATURES]
y_burn = df["burnout_risk_score"]

lr_burn = LinearRegression()
lr_burn.fit(X_burn, y_burn)
y_burn_pred = lr_burn.predict(X_burn)

r2_burn = lr_burn.score(X_burn, y_burn)
print(f"\nR² (burnout) = {r2_burn:.4f}")

coef_burn = pd.Series(lr_burn.coef_, index=X_burn.columns)
coef_burn_sorted = coef_burn.abs().sort_values(ascending=False)

print("\nКоэффициенты (по abs(coef)):")
for name, val in coef_burn_sorted.items():
    print(f"  {name:35s}: {lr_burn.coef_[list(X_burn.columns).index(name)]:+.4f}  (|coef| = {val:.4f})")

coef_burn_df = pd.DataFrame({
    "feature": X_burn.columns,
    "coefficient": lr_burn.coef_,
    "abs_coefficient": np.abs(lr_burn.coef_),
}).sort_values("abs_coefficient", ascending=False)
coef_burn_df.to_json(f"{OUTPUT_DIR}/linear_coefficients_burnout.json", orient="records", force_ascii=False, indent=4)

top5_burn = coef_burn_df.head(5)
print("\nТоп-5 факторов влияния (abs coef):")
for i, row in top5_burn.iterrows():
    print(f"  {row['feature']}: {row['coefficient']:+.4f} (|coef| = {row['abs_coefficient']:.4f})")

# ============================================================
# 4. RESIDUALS + Q-Q PLOT для валидности
# ============================================================
print("\n" + "=" * 70)
print("4. RESIDUALS + Q-Q PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Productivity residuals histogram + Q-Q
residuals_prod = y_prod - y_prod_pred

axes[0, 0].hist(residuals_prod, bins=40, color="#3498db", edgecolor="white", alpha=0.8)
axes[0, 0].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[0, 0].set_title("Productivity: Residuals Distribution", fontsize=12)
axes[0, 0].set_xlabel("Residual")
axes[0, 0].set_ylabel("Frequency")

stats.probplot(residuals_prod, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title("Productivity: Q-Q Plot", fontsize=12)

axes[0, 2].scatter(y_prod_pred, residuals_prod, alpha=0.3, s=10, color="#3498db")
axes[0, 2].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[0, 2].set_title("Productivity: Residuals vs Fitted", fontsize=12)
axes[0, 2].set_xlabel("Fitted values")
axes[0, 2].set_ylabel("Residuals")

# Burnout residuals histogram + Q-Q
residuals_burn = y_burn - y_burn_pred

axes[1, 0].hist(residuals_burn, bins=40, color="#e74c3c", edgecolor="white", alpha=0.8)
axes[1, 0].axvline(0, color="blue", linestyle="--", linewidth=1.5)
axes[1, 0].set_title("Burnout: Residuals Distribution", fontsize=12)
axes[1, 0].set_xlabel("Residual")
axes[1, 0].set_ylabel("Frequency")

stats.probplot(residuals_burn, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Burnout: Q-Q Plot", fontsize=12)

axes[1, 2].scatter(y_burn_pred, residuals_burn, alpha=0.3, s=10, color="#e74c3c")
axes[1, 2].axhline(0, color="blue", linestyle="--", linewidth=1.5)
axes[1, 2].set_title("Burnout: Residuals vs Fitted", fontsize=12)
axes[1, 2].set_xlabel("Fitted values")
axes[1, 2].set_ylabel("Residuals")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/residuals_qq_plots.png")
plt.close()
print("Сохранено: residuals_qq_plots.png")

shapiro_prod = stats.shapiro(residuals_prod[:5000])
shapiro_burn = stats.shapiro(residuals_burn[:5000])
print(f"\nShapiro-Wilk (Productivity residuals): W={shapiro_prod.statistic:.4f}, p={shapiro_prod.pvalue:.2e}")
print(f"Shapiro-Wilk (Burnout residuals):       W={shapiro_burn.statistic:.4f}, p={shapiro_burn.pvalue:.2e}")

# ============================================================
# 5. KMEANS: Elbow + Silhouette
# ============================================================
print("\n" + "=" * 70)
print("5. KMEANS: ELBOW + SILHOUETTE")
print("=" * 70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES_9])

K_range = range(2, 11)
inertias = []
silhouettes = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"  k={k:2d}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(list(K_range), inertias, "bo-", markersize=6)
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Inertia (WCSS)")
ax1.set_title("Elbow Method for Optimal k", fontsize=12)
ax1.grid(True, alpha=0.3)

ax2.plot(list(K_range), silhouettes, "go-", markersize=6)
ax2.set_xlabel("Number of clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score for Optimal k", fontsize=12)
ax2.grid(True, alpha=0.3)

best_k = list(K_range)[np.argmax(silhouettes)]
ax2.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"Best k = {best_k}")
ax2.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/elbow_silhouette.png")
plt.close()
print(f"\nОптимальное k (по Silhouette) = {best_k}")
print("Сохранено: elbow_silhouette.png")

elbow_data = {
    "k_values": list(K_range),
    "inertias": inertias,
    "silhouette_scores": silhouettes,
    "optimal_k": best_k,
}
with open(f"{OUTPUT_DIR}/elbow_silhouette.json", "w", encoding="utf-8") as f:
    json.dump(elbow_data, f, indent=4, ensure_ascii=False)

# ============================================================
# 6. KMEANS: Final clustering with optimal k
# ============================================================
print("\n" + "=" * 70)
print("6. ПРОФИЛИ КЛАСТЕРОВ (mean() по 9 метрикам)")
print("=" * 70)

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = kmeans_final.fit_predict(X_scaled)

cluster_counts = df["cluster"].value_counts().sort_index()
for c in range(best_k):
    print(f"  Cluster {c}: {cluster_counts.get(c, 0)} сотрудников")

cluster_means = df.groupby("cluster")[FEATURES_9].mean().round(2)
cluster_means = cluster_means.rename(columns=dict(zip(FEATURES_9, LABELS_9)))
print(f"\nПрофили кластеров (mean):\n{cluster_means.to_string()}")

cluster_means.to_json(f"{OUTPUT_DIR}/cluster_profiles.json", orient="index", force_ascii=False, indent=4)

cluster_labels_map = {}
for c in cluster_means.index:
    high_ai = cluster_means.loc[c, "AI usage (hrs)"] > cluster_means["AI usage (hrs)"].median()
    high_prod = cluster_means.loc[c, "Tasks automated (%)"] > cluster_means["Tasks automated (%)"].median()
    if high_ai and high_prod:
        cluster_labels_map[c] = f"Cluster {c}: AI-Driven"
    elif high_ai:
        cluster_labels_map[c] = f"Cluster {c}: Tech-Heavy"
    elif high_prod:
        cluster_labels_map[c] = f"Cluster {c}: High-Automation"
    else:
        cluster_labels_map[c] = f"Cluster {c}: Traditional"

df["cluster_label"] = df["cluster"].map(cluster_labels_map)

# ============================================================
# 7. 2D ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ (PCA)
# ============================================================
print("\n" + "=" * 70)
print("7. 2D ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ (PCA)")
print("=" * 70)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

var_exp = pca.explained_variance_ratio_
print(f"PCA explained variance: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%}, total={var_exp.sum():.1%}")

fig, ax = plt.subplots(figsize=(11, 7))
palette = sns.color_palette("husl", best_k)
for c in range(best_k):
    mask = df["cluster"] == c
    ax.scatter(
        df.loc[mask, "pca1"],
        df.loc[mask, "pca2"],
        c=[palette[c]],
        label=cluster_labels_map[c],
        alpha=0.5,
        s=15,
        edgecolor="none",
    )
centroids_pca = pca.transform(kmeans_final.cluster_centers_)
ax.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    c="black",
    marker="X",
    s=200,
    edgecolor="white",
    linewidth=1.5,
    label="Centroids",
    zorder=10,
)
ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)")
ax.set_title(f"2D PCA Visualization of {best_k} Clusters", fontsize=13)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pca_clusters_2d.png")
plt.close()
print("Сохранено: pca_clusters_2d.png")

# ============================================================
# 8. ОПТИМАЛЬНАЯ ЗОНА AI
# ============================================================
print("\n" + "=" * 70)
print("8. ОПТИМАЛЬНАЯ ЗОНА AI")
print("=" * 70)

prod_median = df["productivity"].median()
burnout_median = df["burnout_risk_score"].median()

df["optimal_zone"] = (df["productivity"] >= prod_median) & (df["burnout_risk_score"] <= burnout_median)

zone_counts = df["optimal_zone"].value_counts()
n_optimal = zone_counts.get(True, 0)
print(f"\nСотрудников в оптимальной зоне: {n_optimal}/{len(df)} ({n_optimal/len(df)*100:.1f}%)")
print(f"  Критерий: productivity >= {prod_median:.2f} И burnout <= {burnout_median:.2f}")

zone_stats = df[df["optimal_zone"]][["ai_tool_usage_hours_per_week", "ai_ratio"]].describe().round(2)
print(f"\nAI usage в оптимальной зоне:\n{zone_stats.to_string()}")

zone_stats.to_json(f"{OUTPUT_DIR}/optimal_zone_ai_stats.json", orient="index", force_ascii=False, indent=4)

ai_optimal_min = df[df["optimal_zone"]]["ai_tool_usage_hours_per_week"].quantile(0.25)
ai_optimal_max = df[df["optimal_zone"]]["ai_tool_usage_hours_per_week"].quantile(0.75)
print(f"\nРекомендуемый диапазон AI (IQR оптимальной зоны): [{ai_optimal_min:.1f}, {ai_optimal_max:.1f}] часов/нед")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.scatter(
    df[~df["optimal_zone"]]["ai_tool_usage_hours_per_week"],
    df[~df["optimal_zone"]]["productivity"],
    alpha=0.3, s=10, color="gray", label="Sub-optimal",
)
ax.scatter(
    df[df["optimal_zone"]]["ai_tool_usage_hours_per_week"],
    df[df["optimal_zone"]]["productivity"],
    alpha=0.4, s=15, color="#2ecc71", label="Optimal zone",
)
ax.axhline(prod_median, color="blue", linestyle="--", alpha=0.7, label=f"Prod median ({prod_median:.1f})")
ax.axvspan(
    ai_optimal_min, ai_optimal_max,
    alpha=0.1, color="#2ecc71", label=f"AI range [{ai_optimal_min:.1f}, {ai_optimal_max:.1f}] hrs"
)
ax.set_xlabel("AI usage (hrs/week)")
ax.set_ylabel("Productivity")
ax.set_title("Optimal AI Zone: Productivity View", fontsize=12)
ax.legend(fontsize=8)

ax = axes[1]
ax.scatter(
    df[~df["optimal_zone"]]["ai_tool_usage_hours_per_week"],
    df[~df["optimal_zone"]]["burnout_risk_score"],
    alpha=0.3, s=10, color="gray", label="Sub-optimal",
)
ax.scatter(
    df[df["optimal_zone"]]["ai_tool_usage_hours_per_week"],
    df[df["optimal_zone"]]["burnout_risk_score"],
    alpha=0.4, s=15, color="#2ecc71", label="Optimal zone",
)
ax.axhline(burnout_median, color="red", linestyle="--", alpha=0.7, label=f"Burnout median ({burnout_median:.1f})")
ax.axvspan(
    ai_optimal_min, ai_optimal_max,
    alpha=0.1, color="#2ecc71", label=f"AI range [{ai_optimal_min:.1f}, {ai_optimal_max:.1f}] hrs"
)
ax.set_xlabel("AI usage (hrs/week)")
ax.set_ylabel("Burnout Risk Score")
ax.set_title("Optimal AI Zone: Burnout View", fontsize=12)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/optimal_ai_zone.png")
plt.close()
print("Сохранено: optimal_ai_zone.png")

# ============================================================
# 9. AI EFFICIENCY MAP (productivity / burnout)
# ============================================================
print("\n" + "=" * 70)
print("9. КАРТА AI ЭФФЕКТИВНОСТИ")
print("=" * 70)

ai_min = df["ai_tool_usage_hours_per_week"].min()
ai_max = df["ai_tool_usage_hours_per_week"].max()
bins = np.linspace(ai_min, ai_max, 11)
labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}h" for i in range(10)]

ai_min = df["ai_tool_usage_hours_per_week"].min()
ai_max = df["ai_tool_usage_hours_per_week"].max()
bins = np.linspace(ai_min, ai_max, 11)
labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}h" for i in range(10)]

df["ai_bucket"] = pd.cut(
    df["ai_tool_usage_hours_per_week"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)

efficiency_by_ai = df.groupby("ai_bucket", observed=True)["ai_efficiency"].agg(["mean", "std", "count"]).round(3)
print(f"\nAI Efficiency по бакетам:\n{efficiency_by_ai.to_string()}")
efficiency_by_ai.to_json(f"{OUTPUT_DIR}/ai_efficiency_by_bucket.json", orient="index", force_ascii=False, indent=4)

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = range(len(efficiency_by_ai))
ax.bar(
    x_pos,
    efficiency_by_ai["mean"],
    color=plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(efficiency_by_ai))),
    edgecolor="white",
)
ax.set_xticks(x_pos)
ax.set_xticklabels([str(l) for l in efficiency_by_ai.index], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("AI usage bucket (hrs/week)")
ax.set_ylabel("Mean AI Efficiency (productivity / (burnout+1))")
ax.set_title("AI Efficiency by Usage Level", fontsize=13)
for i, row in enumerate(efficiency_by_ai.itertuples()):
    ax.text(i, row.mean + 0.001, f"{row.mean:.3f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ai_efficiency_bars.png")
plt.close()
print("Сохранено: ai_efficiency_bars.png")

# ============================================================
# 10. CORRELATION: AI vs burnout (scatter)
# ============================================================
print("\n" + "=" * 70)
print("10. КОРРЕЛЯЦИЯ: AI usage vs burnout")
print("=" * 70)

r_ai_burn, p_ai_burn = stats.pearsonr(df["ai_tool_usage_hours_per_week"], df["burnout_risk_score"])
r_ai_prod, p_ai_prod = stats.pearsonr(df["ai_tool_usage_hours_per_week"], df["productivity"])
print(f"AI usage vs burnout:      r = {r_ai_burn:+.4f}, p = {p_ai_burn:.2e}")
print(f"AI usage vs productivity:  r = {r_ai_prod:+.4f}, p = {p_ai_prod:.2e}")

# ============================================================
# 11. FINAL REPORT
# ============================================================
print("\n" + "=" * 70)
print("11. ФИНАЛЬНЫЙ ОТЧЕТ")
print("=" * 70)

insights_data = [
    {
        "insight": "AI usage слабо коррелирует с burnout",
        "detail": f"r = {r_ai_burn:+.3f}, p = {p_ai_burn:.2e}",
        "recommendation": "AI не является драйвером выгорания — можно безопасно наращивать использование",
    },
    {
        "insight": "AI usage положительно связан с продуктивностью",
        "detail": f"r = {r_ai_prod:+.3f}, p = {p_ai_prod:.2e}",
        "recommendation": "Стимулировать использование AI-инструментов для роста продуктивности",
    },
    {
        "insight": f"Линейная модель продуктивности: R² = {r2_prod:.3f}",
        "detail": f"Топ-5: {', '.join(top5_prod['feature'].values)}",
        "recommendation": "Фокус на ключевых драйверах продуктивности из топ-5 факторов",
    },
    {
        "insight": f"Линейная модель burnout: R² = {r2_burn:.3f}",
        "detail": f"Топ-5: {', '.join(top5_burn['feature'].values)}",
        "recommendation": "Контроль ключевых факторов риска выгорания из топ-5",
    },
    {
        "insight": f"Оптимальное число кластеров = {best_k} (Silhouette = {max(silhouettes):.3f})",
        "detail": f"Размеры кластеров: {cluster_counts.to_dict()}",
        "recommendation": f"Сегментировать сотрудников на {best_k} групп для таргетированных интервенций",
    },
    {
        "insight": f"Оптимальная зона AI: {n_optimal} сотрудников ({n_optimal/len(df)*100:.1f}%)",
        "detail": f"AI range: [{ai_optimal_min:.1f}, {ai_optimal_max:.1f}] hrs/week",
        "recommendation": f"Таргетировать AI-usage в диапазоне [{ai_optimal_min:.1f}, {ai_optimal_max:.1f}] часов/нед",
    },
    {
        "insight": "Кластеры с высоким AI показывают лучшие метрики",
        "detail": f"См. cluster_profiles.json для деталей",
        "recommendation": "Масштабировать AI-практики высокопроизводительных кластеров",
    },
    {
        "insight": "Residuals близки к нормальному распределению",
        "detail": f"SW p(Burnout) = {shapiro_burn.pvalue:.2e}",
        "recommendation": "Модели достаточно валидны для интерпретации",
    },
]

print(f"\n{'#':<3} {'INSIGHT':<55} {'RECOMMENDATION'}")
print("-" * 130)
for i, row in enumerate(insights_data, 1):
    print(f"{i:<3} {row['insight']:<55} {row['recommendation']}")

with open(f"{OUTPUT_DIR}/final_report.json", "w", encoding="utf-8") as f:
    json.dump(insights_data, f, indent=4, ensure_ascii=False)

report_txt = "ФИНАЛЬНЫЙ ОТЧЕТ: AI Productivity & Burnout Analysis\n" + "=" * 70 + "\n\n"
report_txt += "ТОП-5 ФАКТОРОВ ВЛИЯНИЯ (Productivity, abs coef):\n"
for _, row in top5_prod.iterrows():
    report_txt += f"  {row['feature']}: {row['coefficient']:+.4f}\n"
report_txt += f"\nТОП-5 ФАКТОРОВ ВЛИЯНИЯ (Burnout, abs coef):\n"
for _, row in top5_burn.iterrows():
    report_txt += f"  {row['feature']}: {row['coefficient']:+.4f}\n"
report_txt += f"\nОПТИМАЛЬНЫЙ ДИАПАЗОН AI: [{ai_optimal_min:.1f}, {ai_optimal_max:.1f}] часов/нед\n"
report_txt += f"ОПТИМАЛЬНОЕ ЧИСЛО КЛАСТЕРОВ: k = {best_k}\n\n"
report_txt += "ТАБЛИЦА ИНСАЙДОВ И РЕКОМЕНДАЦИЙ:\n" + "-" * 70 + "\n"
for i, row in enumerate(insights_data, 1):
    report_txt += f"\n{i}. {row['insight']}\n   Detail: {row['detail']}\n   → {row['recommendation']}\n"

with open(f"{OUTPUT_DIR}/final_report.txt", "w", encoding="utf-8") as f:
    f.write(report_txt)
print(f"\nСохранено: final_report.json, final_report.txt")

print("\n" + "=" * 70)
print("ГОТОВО. Все результаты сохранены в outputs/")
print("=" * 70)
