
from main import small_main

import os
import pandas as pd
import shutil


# 路径设置
base_path = "../train_base_model/result"
seed_folders = [f"np_seed{i}" for i in range(10)]

# 定义指标信息
metrics = ["MAE", "MedAE", "MRE", "MedRE", "R2", "RMSE"]
better_high = {"R2"}  # 越大越好的指标

def select_best_model():
    all_data = []

    # 读取数据
    for folder in seed_folders:
        file_path = os.path.join(base_path, folder, '_model', "result.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)
            df.columns = ["batch_size", "lr", "init_method", "epoch"] + metrics
            df["folder"] = folder
            all_data.append(df)

    # 合并所有数据
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)

        # 统计每个文件夹在每个指标中是否为最优
        best_counts = pd.Series(0, index=df_all["folder"].unique())
        for metric in metrics:
            if metric in better_high:
                best_value = df_all[metric].max()
            else:
                best_value = df_all[metric].min()
            best_folders = df_all[df_all[metric] == best_value]["folder"]
            for folder in best_folders:
                best_counts[folder] += 1

        # 找到最佳次数最多的文件夹
        best_folder = best_counts.idxmax()
        best_score = best_counts.max()
        print(best_folder, best_score)
        return eval(best_folder[-1])
    else:
        print("all data is empty!")





def main():
    # 找出性能最好的基模型
    best_seed = select_best_model()
    # 微调
    small_main(best_seed)


main()

