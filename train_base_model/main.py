import os
import sys
import time
import numpy as np
from keras.models import load_model

sys.path.append("../")

from train_base_model.cross_validate import cross_validate, LossHistory
from train_base_model.build import build_DNN


# from train_small_data.main import small_main

import utils

data_path = "../data/"
pro_path = "../"
seed = 0


def search():
    batch_size_list = [16, 32, 64, 128]  # 每次送入的数据量
    lr_list = [0.001, 0.005, 0.01]  # 学习率
    init_list = ["glorot_normal", "glorot_uniform", "random_normal"] # 参数初始化方式
    epochs_list = [10, 20, 40, 80, 160, 320]  # 不同训练epochs

    combine_list = []
    for batch_size in batch_size_list:
        for lr in lr_list:
            for init in init_list:
                combine_list.append([batch_size, lr, init, epochs_list])

    begin_time = time.perf_counter()

    # 读取数据
    X_train = utils.read_csv(data_path + "gen/np_seed" + str(seed) + "/" + "X_train.csv", header=None)
    y_train = utils.read_csv(data_path + "gen/np_seed" + str(seed) + "/" + "y_train.csv", header=None)[0]
    X_val = utils.read_csv(data_path + "gen/np_seed" + str(seed) + "/" + "X_val.csv", header=None)
    y_val = utils.read_csv(data_path + "gen/np_seed" + str(seed) + "/" + "y_val.csv", header=None)[0]

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    # 结果汇总表
    # header_all = ["batch_size", "lr", "init", "cur_epochs", "MAE", "MedAE", "MRE", "MedRE", "R_square", "RMSE"]
    path = pro_path + "train_base_model/result/np_seed" + str(seed) + "/"
    result_all = []
    cnt = 0
    for i in range(len(combine_list)):
        batch_size, lr, init, epochs_list_e = combine_list[i]

        start = time.perf_counter()  # 开始计时

        cnt += 1
        print("-" * 60)
        print("当前训练进度：", cnt, "/", len(combine_list))

        # DNN训练
        for epoch in epochs_list_e:

            result = cross_validate(X_train, X_val, y_train, y_val, batch_size, lr, init, epoch, path)
            # 结果输出并写入文件
            for e in result: result_all.append(e)
            utils.write_csv(path + "val/", "result_val_all.csv", result_all)
            del result
            print("本次训练用时: " + utils.cost_time(start, time.perf_counter()))



    # 选择在验证集结果最好的一组参数，得到其训练模型
    result_all.sort(key=lambda x: x[4])
    print("验证集结果最好对应参数 & 结果：", result_all[0])
    batch_size, lr, init, cur_epochs = result_all[0][0], result_all[0][1], result_all[0][2], result_all[0][3]
    model = build_DNN(X_train.shape[1], lr, init)
    history = LossHistory()
    model.fit(X_train, y_train, epochs=cur_epochs, batch_size=batch_size, verbose=2, callbacks=[history])
    predict_val = np.array(model(X_val))[:, 0]
    index = [batch_size, lr, init, cur_epochs] + utils.cal_index(y_val, predict_val)

    result_path = path + "_model/"
    utils.write_csv(result_path, "result.csv", np.array(index, dtype=object).reshape(1, -1))
    history.loss_plot('epoch', result_path)
    model.save(result_path + "base_model.h5")

    print("总用时: " + utils.cost_time(begin_time, time.perf_counter()))


def main(seed):

    # 预训练
    search()
    # 微调
    # small_main(seed)




start_time = time.time()

for i in range(0, 10):
    seed = i
    ## 训练模型
    main(seed)


print("finish")

end_time = time.time()

elapsed_time  = end_time - start_time

days = int(elapsed_time // (24 * 3600))
elapsed_time = elapsed_time % (24 * 3600)
hours = int(elapsed_time // 3600)
elapsed_time %= 3600
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)


print(f"总用时：{days}天 {hours}小时 {minutes}分 {seconds}秒")
current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(current_time)