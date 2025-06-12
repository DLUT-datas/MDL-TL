import os
import pandas as pd
import numpy as np

import utils



def collect(dataset_list, see):
    for dataset in dataset_list:
        # os.makedirs(dataset, exist_ok=True)
        # 10个种子的文件夹名为 'seed0', 'seed1', ..., 'seed9'
        seed = f"np_seed{see}"

        # 存储所有种子的预测结果
        all_preds = []
        all_trues = []
        folder_name = dataset
        data_path = '../../data/'
        # target_path = '../../data/target'
        smiles_rt_file_path = os.path.join(data_path, 'smiles_rt', f'{folder_name}.txt')
        # 读取测试数据索引路径
        test_data_idx_path = os.path.join(data_path, 'target', folder_name, 'split_index', seed, 'test_index.csv')
        # 读取smiles文件
        smiles = pd.read_csv(smiles_rt_file_path, sep='\t')['smiles']
        # 读取测试集索引
        test_data_idx = pd.read_csv(test_data_idx_path, header=None)
        # 获取第一行作为索引列表
        indices = test_data_idx.iloc[0].tolist()
        # 获取测试数据
        test_smiles = smiles.iloc[indices]

        # 构造路径
        folder_path = seed
        csv_file_path = os.path.join(folder_path, dataset, "pred_result.csv")

        if os.path.exists(csv_file_path):
            # 读取CSV文件
            df = pd.read_csv(csv_file_path)
        else:
            print(f"警告: {csv_file_path} 不存在")


        # 创建一个新的 DataFrame 用于汇总预测结果
        result_df = pd.DataFrame({
            'SMILES': test_smiles.values,
            'pred': df['pred'].values,
            'true': df['true'].values
        })
        # 计算性能指标
        header = ['MAE','MedAE','MRE','MedRE','R_square','RMSE']
        index_result = [header]
        index = utils.cal_index(df['true'].values, df['pred'].values)
        index_result.append(index)

        collect_pred_path = f'./collect/{seed}/pred'
        collect_index_path = f'./collect/{seed}/index'


        os.makedirs(collect_pred_path, exist_ok=True)
        os.makedirs(collect_index_path, exist_ok=True)

        # 写出性能指标
        utils.write_csv(collect_index_path, f'{folder_name}_index_results.csv', index_result)

        # 写出预测结果
        result_df.to_csv(os.path.join(collect_pred_path, f'{folder_name}_results.csv'), index=False)

        print("结果已写入: ", f'{folder_name}_results.csv')

def main():
    # 要收集的种子
    seed = 0
    # 要收集的数据集
    dataset_list = os.listdir('../data/target')

    print(dataset_list)
    collect(dataset_list, seed)

main()