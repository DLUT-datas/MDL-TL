import os
import numpy as np

import sys
sys.path.append("../")
import utils

data_path = "../data/"
pro_path = "../"

if __name__ == "__main__":

    file_list = os.listdir(data_path + "target/")

    input_dirs = []
    for file in file_list:
        if len(file.split('.')) == 2:  # 说明是文件
            continue
        input_dirs.append(file)

    print(input_dirs)
    print(len(input_dirs))

    np_list = [i for i in range(10)]
    result = utils.read_csv(pro_path + "train_small_data/result/np_seed" + str(np_list[0]) + "/result.csv", header=None)
    result_np_seed = [result[:, 0], ["0"] + result[1:, 1].tolist()]
    for u in range(1, len(result)):
        for v in range(1, len(result[0])):
            result[u][v] = float(result[u][v])

    cnt = len(np_list)

    for i in range(1, len(np_list)):
        seed = np_list[i]
        path = pro_path + "train_small_data/result/np_seed" + str(seed) + "/"
        tmp = utils.read_csv(path + "result.csv", header=None)
        result_np_seed.append([str(seed)] + tmp[1:, 1].tolist())
        for u in range(1, len(result)):
            for v in range(1, len(result[0])):
                result[u][v] += float(tmp[u][v])

    for u in range(1, len(result)):
        for v in range(1, len(result[0])):
            result[u][v] /= cnt
            result[u][v] = round(result[u][v], 4)
    utils.write_csv(pro_path + "train_small_data/result/", "result_avg.csv", result)
