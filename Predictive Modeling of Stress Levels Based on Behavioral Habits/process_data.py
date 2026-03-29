import pandas as pd
import numpy as np


def process_wellness_data(input_file, output_file):
    df = pd.read_csv(input_file)

    df["total_tech_usage"] = df["phone_usage_hours"] + df["laptop_usage_hours"]

    passive_sum = df["social_media_hours"] + df["entertainment_hours"]
    active_sum = df["work_related_hours"] + df["gaming_hours"]

    conditions = [(active_sum > passive_sum), (passive_sum > active_sum)]
    choices = ["Active/High-Intensity Usage", "Passive Usage"]

    df["usage_category"] = np.select(conditions, choices, default="Balanced")

    initial_count = len(df)
    df = df[df["sleep_duration_hours"] <= 24]

    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"检测到异常：已删除 {removed_count} 条睡眠时长异常的数据。")

    df.to_csv(output_file, index=False)
    print(f"处理完成！处理后的数据已保存至: {output_file}")
    return df


if __name__ == "__main__":
    input_path = "Tech_Use_Stress_Wellness.csv"
    output_path = "Processed_Tech_Stress_Data.csv"
    processed_df = process_wellness_data(input_path, output_path)

    print(
        processed_df[
            ["total_tech_usage", "usage_category", "sleep_duration_hours"]
        ].head()
    )
