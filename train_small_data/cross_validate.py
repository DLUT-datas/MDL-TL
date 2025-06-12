import numpy as np
import keras
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import sys
import os
import gc
import pandas as pd
# sys.path.append("E:\Coding\Python\wxx\others\explore")
sys.path.append("../")
from train_small_data.build import build_DNN
import warnings

import utils

warnings.filterwarnings("ignore")


def cross_validate(X_train, y_train, X_val, y_val, X_test, y_test, model_path, result_path, combine_list, inner_CV_times, pred_path):

    # 使用 X_non_test、y_non_test 寻找最优参数，之后在 X_test、y_test 上测试结果
    result_all = []  # 存储 所有参数组合 combine_list 测试 inner_CV_times 的平均结果，[batch_size, lr, epochs, ...]
    for i in range(len(combine_list)):
        batch_size, lr, epochs_list = combine_list[i]

        # print("结果保存路径为: ", result_path)

        result = {}  # 是个map, (key: epochs, val: []), val是个二维列表，存储 inner_CV_times 次结果
        for j in range(inner_CV_times):
            print("当前搜索第" + str(i) + "大组参数...")

            # model = build_DNN(lr, model_path)
            # history = LossHistory()

            last_epochs = 0  # 类似于累积训练
            for k in range(len(epochs_list)):
                cur_epochs = epochs_list[k]
                model = build_DNN(lr, model_path)
                history = LossHistory()
                print("当前搜索batch_size = " + str(batch_size) + ", lr = " + str(lr) + ", epochs = " + str(cur_epochs))
                model.fit(X_train, y_train, epochs=cur_epochs, batch_size=batch_size, verbose=0,
                          callbacks=[history])
                predict_val = np.array(model(X_val))[:, 0]
                # last_epochs = cur_epochs

                index = utils.cal_index(y_val, predict_val)
                if cur_epochs not in result:
                    result[cur_epochs] = [index]
                else:
                    result[cur_epochs].append(index)
                del model, history
                gc.collect()
        for epochs in epochs_list:
            epochs_result = result[epochs]
            result_avg = []
            for u in range(len(epochs_result[0])):  # 遍历所有列
                s = 0
                for v in range(inner_CV_times):  # 遍历所有行
                    s += epochs_result[v][u]
                result_avg.append(round(s / inner_CV_times, 4))
            result_all.append([batch_size, lr, epochs] + result_avg)

    utils.write_csv(result_path, "result_all_params.csv", result_all)
    # 选择最好的一组参数在 X_test、y_test 上测试结果
    result_all.sort(key=lambda x: x[3])
    best_bs, best_lr, best_epochs = result_all[0][0], result_all[0][1], result_all[0][2]

    model = build_DNN(best_lr, model_path)
    history = LossHistory()
    model.fit(X_train, y_train, epochs=best_epochs, batch_size=best_bs, verbose=0, callbacks=[history])
    predicts = np.array(model(X_test))[:, 0]
    # 写出预测结果

    pred_res = pd.DataFrame({
        'pred': predicts,
        'true': y_test,
    })

    # pred_res_path = os.path.join('..', 'train_small_data', 'pred', f'np_seed{seed}')
    os.makedirs(pred_path, exist_ok=True)
    pred_res_file_path = os.path.join(pred_path, 'pred_result.csv')
    # utils.write_csv(result_path, 'pred_result.csv', )

    pred_res.to_csv(pred_res_file_path, index=False)

    index = utils.cal_index(y_test, predicts)
    utils.write_csv(result_path, "result_best.csv",
                  np.array([best_bs, best_lr, best_epochs] + index, dtype=object).reshape(1, -1))
    res_path = os.path.join(result_path, '_model')
    os.makedirs(res_path, exist_ok=True)
    model.save(os.path.join(result_path, "final_model.h5"))
    del model, history, predicts
    gc.collect()
    return [best_bs, best_lr, best_epochs] + index


class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = {'batch': [], 'epoch': []}

    # def on_train_begin(self, logs={}):
    #     self.losses = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))

    def loss_plot(self, loss_type, path):  # path为保存路径

        path_image = path + "loss.png"

        iters = range(len(self.losses[loss_type]))

        # 保存loss值
        utils.write_csv(path, "loss.csv", np.array(self.losses[loss_type]).reshape(1, -1))

        # 保存loss曲线
        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.title(path.split("/")[-2])
        plt.savefig(path_image)
        plt.close("all")
