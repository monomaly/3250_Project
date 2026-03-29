import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

df = pd.read_csv("Tech_Use_Stress_Wellness.csv")
sns.set(style="whitegrid")

grouped_stats = df.groupby("uses_wellness_apps")[
    ["sleep_quality", "mood_rating", "mindfulness_minutes_per_day"]
].mean()
print("--- 不同应用使用情况下的指标均值 ---")
print(grouped_stats)

users = df[df["uses_wellness_apps"] == True]
non_users = df[df["uses_wellness_apps"] == False]

t_sleep, p_sleep = stats.ttest_ind(users["sleep_quality"], non_users["sleep_quality"])
t_mood, p_mood = stats.ttest_ind(users["mood_rating"], non_users["mood_rating"])

print(f"\n睡眠质量差异检验: t={t_sleep:.4f}, p={p_sleep:.4e}")
print(f"情绪评分差异检验: t={t_mood:.4f}, p={p_mood:.4e}")

corr_sleep, p_corr_sleep = stats.pearsonr(
    df["mindfulness_minutes_per_day"], df["sleep_quality"]
)
corr_mood, p_corr_mood = stats.pearsonr(
    df["mindfulness_minutes_per_day"], df["mood_rating"]
)

print(f"\n正念时长与睡眠相关性: r={corr_sleep:.4f}, p={p_corr_sleep:.4e}")
print(f"正念时长与情绪相关性: r={corr_mood:.4f}, p={p_corr_mood:.4e}")

X = df[["uses_wellness_apps", "mindfulness_minutes_per_day"]].astype(float)
X = sm.add_constant(X)

model_sleep = sm.OLS(df["sleep_quality"], X).fit()
model_mood = sm.OLS(df["mood_rating"], X).fit()

print("\n--- 睡眠质量回归模型结果 ---")
print(model_sleep.summary().tables[1])
print("\n--- 情绪评分回归模型结果 ---")
print(model_mood.summary().tables[1])

plt.figure(figsize=(8, 6))
sns.boxplot(x="uses_wellness_apps", y="sleep_quality", data=df)
plt.title("Sleep Quality vs Wellness App Usage")
plt.tight_layout()
plt.savefig("wellness_impact_sleep.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(x="uses_wellness_apps", y="mood_rating", data=df)
plt.title("Mood Rating vs Wellness App Usage")
plt.tight_layout()
plt.savefig("wellness_impact_mood.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(x="mindfulness_minutes_per_day", y="sleep_quality", data=df, alpha=0.3)
plt.title("Mindfulness Duration vs Sleep Quality")
plt.tight_layout()
plt.savefig("wellness_impact_mindfulness_sleep.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(x="mindfulness_minutes_per_day", y="mood_rating", data=df, alpha=0.3)
plt.title("Mindfulness Duration vs Mood Rating")
plt.tight_layout()
plt.savefig("wellness_impact_mindfulness_mood.png")
plt.close()
