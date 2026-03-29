import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


# ============ Data Loading & Exploration ============
def load_and_explore_data(data_path):
    """加载数据并探索性分析"""
    df = pd.read_csv(data_path)

    features = [
        "sleep_quality",
        "sleep_duration_hours",
        "caffeine_intake_mg_per_day",
        "social_media_hours",
        "work_related_hours",
        "gaming_hours",
        "entertainment_hours",
        "phone_usage_hours",
        "laptop_usage_hours",
    ]
    target = "stress_level"

    df_subset = df[features + [target]].dropna()

    print("Dataset Info:")
    print(df_subset.info())
    print("\nTarget Distribution:")
    print(df_subset[target].value_counts().sort_index())

    # Correlation matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        df_subset.corr(), annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 12}
    )
    plt.title("Correlation Matrix of Features and Stress Level", fontsize=16)
    plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    return df_subset


# ============ Model Training ============
def train_stress_model(df_subset):
    """训练随机森林模型并评估"""
    features = [
        "sleep_quality",
        "sleep_duration_hours",
        "caffeine_intake_mg_per_day",
        "social_media_hours",
        "work_related_hours",
        "gaming_hours",
        "entertainment_hours",
        "phone_usage_hours",
        "laptop_usage_hours",
    ]
    target = "stress_level"

    X = df_subset[features]
    y = df_subset[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test_scaled)
    print(f"\n模型整体准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(
        ascending=False
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="magma")
    plt.title("Feature Importance for Stress Level Prediction")
    plt.xlabel("Importance Weight")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="RdPu",
        xticklabels=range(1, 11),
        yticklabels=range(1, 11),
    )
    plt.title("Confusion Matrix: Predicted vs True Stress Level")
    plt.xlabel("Predicted Level")
    plt.ylabel("Actual Level")
    plt.savefig("stress_confusion_matrix.png")
    plt.close()

    return rf_model, importances


if __name__ == "__main__":
    data_path = "Processed_Tech_Stress_Data.csv"

    # Step 1: Load and explore data
    df_subset = load_and_explore_data(data_path)

    # Step 2: Train model
    model, importances = train_stress_model(df_subset)

    print("\n特征重要性排序:")
    print(importances)
