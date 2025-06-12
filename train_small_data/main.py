import numpy as np
import time
import os

import sys
sys.path.append("../")
# sys.path.append("E:\Coding\Python\wxx\others\explore")
from train_small_data.cross_validate import cross_validate

import utils

data_path = "../data/"
pro_path = "../"
seed = 0


def search(X_train, y_train, X_val, y_val, X_test, y_test, result_path, pred_path):
    batch_size_list = [16, 32]  # 每次送入的数据量
    lr_list = [0.0001, 0.0005, 0.001, 0.005]  # 学习率
    epochs_list = [50, 100, 150, 200, 250, 300, 350, 400]  # 不同训练epochs

    # batch_size_list = [16]  # 每次送入的数据量
    # lr_list = [0.0001]  # 学习率
    # epochs_list = [50, 100]  # 不同训练epochs

    begin_time = time.perf_counter()

    combine_list = []
    for batch_size in batch_size_list:
        for lr in lr_list:
            combine_list.append([batch_size, lr, epochs_list])

    del epochs_list, batch_size_list, lr_list

    # 结果汇总表
    result_all = []
    cnt = 0
    outer_CV_times, inner_CV_times = 1, 1
    model_path = pro_path + "train_base_model/result/np_seed" + str(seed) + "/_model/"
    for i in range(outer_CV_times):
        start = time.perf_counter()  # 开始计时

        cnt += 1
        print("-" * 60)
        print("当前训练进度：", cnt, "/", outer_CV_times)

        if not os.path.exists(result_path): os.makedirs(result_path)
        print("结果保存路径为: ", result_path)

        # result是一个列表
        print(f"train_small_data_parameters: {X_train, y_train, X_val, y_val, X_test, y_test, model_path, result_path, combine_list, inner_CV_times}")

        result = cross_validate(X_train, y_train, X_val, y_val, X_test, y_test, model_path, result_path, combine_list,
                                inner_CV_times, pred_path)

        # 结果输出并写入文件
        # print(["best_batch_size", "best_lr", "best_epochs", "MAE", "MedAE", "MRE", "MedRE", "R_square", "RMSE"])
        # print(result)
        print(["MAE", "MedAE", "MRE", "MedRE", "R_square", "RMSE"])
        print(result[3:])
        result_all.append(result)

        print("本次训练用时: " + utils.cost_time(start, time.perf_counter()))

    print("len(result_all)", len(result_all))
    result_avg = []
    for u in range(3, len(result_all[0])):  # 遍历所有列
        s = 0
        for v in range(outer_CV_times):  # 遍历所有行
            s += result_all[v][u]
        result_avg.append(round(s / outer_CV_times, 4))

    del result_all, combine_list
    print("总用时: " + utils.cost_time(begin_time, time.perf_counter()))

    return result_avg


def small_main(see):
    # 声明全局变量
    global seed
    seed = see
    file_list = os.listdir(data_path + "target/")

    # input_dirs = []
    input_dirs = ['negtive_', 'positive_']

    '''
    for file in file_list:
        if len(file.split('.')) == 2:  # 说明是文件
            continue
        input_dirs.append(file)
    '''
    # (0) 'Eawag_XBridgeC18', (1) 'FEM_lipids', (2) 'FEM_long',
    # (3) 'IPB_Halle', (4) 'LIFE_new', (5) 'LIFE_old', (6) 'lipids', (7) 'MassBank1',
    # (8) 'MassBank2', (9) 'MetaboBASE', (10) 'Natural products', (11) 'pesticide',
    # (12) 'RIKEN_PlaSMA', (13) 'UniToyama_Atlantis'
    print(input_dirs)
    print(len(input_dirs))

    header_test = ["dataset", "MAE", "MedAE", "MRE", "MedRE", "R_square", "RMSE"]
    result_test = []
    for file in input_dirs:
        # 读取数据
        X_train = utils.read_csv(data_path + "gen/np_seed" + str(seed) + "/" + file + "/X_train.csv", header=None)
        y_train = utils.read_csv(data_path + "gen/np_seed" + str(seed) + "/" + file + "/y_train.csv", header=None)[0]
        X_val = utils.read_csv(data_path + "gen/np_seed" + str(seed) + "/" + file + "/X_val.csv", header=None)
        y_val = utils.read_csv(data_path + "gen/np_seed" + str(seed) + "/" + file + "/y_val.csv", header=None)[0]
        X_test = utils.read_csv(data_path + "gen/np_seed" + str(seed) + "/" + file + "/X_test.csv", header=None)
        y_test = utils.read_csv(data_path + "gen/np_seed" + str(seed) + "/" + file + "/y_test.csv", header=None)[0]

        X_train, y_train = X_train.astype("float64"), y_train.astype("float64")
        X_val, y_val = X_val.astype("float64"), y_val.astype("float64")
        X_test, y_test = X_test.astype("float64"), y_test.astype("float64")

        res = search(X_train, y_train, X_val, y_val, X_test, y_test,
                     pro_path + "train_small_data/result/np_seed" + str(seed) + "/" + file + "/",
                     pro_path + "train_small_data/pred/np_seed" + str(seed) + "/" + file + "/")
        res.insert(0, file)
        result_test.append(res)

        del X_train, y_train, X_val, y_val, X_test, y_test

    # order_index = [5, 2, 1, 0, 4, 3, 13, 9, 12, 7, 8, 11, 10, 6]
    # result_order = []
    # for index in order_indfex:
    #     result_order.append(result_test[index])
    result_order = result_test
    result_order.insert(0, header_test)
    utils.write_csv(pro_path + "train_small_data/result/np_seed" + str(seed) + "/", "result.csv", result_order)

# if __name__ == '__main__':
#     small_main(6)
#     pass