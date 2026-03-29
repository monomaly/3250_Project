import pandas as pd
import numpy as np


def process_wellness_data(input_file, output_file):
    # 1. 加载数据
    df = pd.read_csv(input_file)

    # 2. 计算 Total Tech Usage
    # 创建新列，合并手机与电脑使用时长
    df["total_tech_usage"] = df["phone_usage_hours"] + df["laptop_usage_hours"]

    # 3. 划分 Passive vs. Active 类别
    # 定义中间变量进行逻辑比较
    passive_sum = df["social_media_hours"] + df["entertainment_hours"]
    active_sum = df["work_related_hours"] + df["gaming_hours"]

    # 使用 np.select 处理多条件分支，效率高于 apply(lambda)
    conditions = [(active_sum > passive_sum), (passive_sum > active_sum)]
    choices = ["Active/High-Intensity Usage", "Passive Usage"]

    # 默认值为 Balanced
    df["usage_category"] = np.select(conditions, choices, default="Balanced")

    # 4. 异常值处理
    # 剔除不合理数据（如睡眠时长 > 24h）
    initial_count = len(df)
    df = df[df["sleep_duration_hours"] <= 24]

    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"检测到异常：已删除 {removed_count} 条睡眠时长异常的数据。")

    # 5. 保存结果
    df.to_csv(output_file, index=False)
    print(f"处理完成！处理后的数据已保存至: {output_file}")
    return df


if __name__ == "__main__":
    # 执行脚本
    input_path = "Tech_Use_Stress_Wellness.csv"
    output_path = "Processed_Tech_Stress_Data.csv"
    processed_df = process_wellness_data(input_path, output_path)

    # 打印前 5 行查看结果
    print(
        processed_df[
            ["total_tech_usage", "usage_category", "sleep_duration_hours"]
        ].head()
    )
