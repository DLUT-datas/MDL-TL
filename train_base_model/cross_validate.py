import numpy as np
import keras
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("../")
# sys.path.append("E:\Coding\Python\wxx\others\explore")
from train_base_model.build import build_DNN

import utils


def cross_validate(X_train, X_val, y_train, y_val, batch_size, lr, init, epoch, path):
    model = build_DNN(X_train.shape[1], lr, init)
    history = LossHistory()

    # print(model.summary())
    print(f"history:{history}")

    print(f"history_type:{type(history)}")

    res = []

    cur_epochs = epoch
    item = [batch_size, lr, init, cur_epochs]
    # print(f"type_X_train:{type(X_train)}")
    # print(f"type_y_train:{type(y_train)}")
    # print(f"dtype_X_train:{X_train.dtype}")
    # print(f"dtype_y_train:{y_train.dtype}")
    # print(f"type_epochs:{type(cur_epochs - last_epochs)}")
    # print(f"type_batch_size:{type(batch_size)}")
    # print(f"callbacks:{type([history])}")

    model.fit(X_train, y_train, epochs=cur_epochs, batch_size=batch_size, verbose=2,
              callbacks=[history])
    predict_val = np.array(model(X_val))[:, 0]
    # last_epochs = cur_epochs

    index = utils.cal_index(y_val, predict_val)
    for e in index: item.append(e)
    res.append(item)

    result_path = path + "/val/model_" + init + "_" + str(batch_size) + "_" + str(lr) + "_" + str(cur_epochs) + "/"
    if not os.path.exists(result_path): os.makedirs(result_path)
    print("验证集结果保存路径为: ", result_path)
    print(["batch_size", "lr", "init", "cur_epochs", "MAE", "MedAE", "MRE", "MedRE", "R_square", "RMSE"])
    print(item)

    utils.write_csv(result_path, "result_val.csv", np.array(index, dtype=object).reshape(1, -1))
    history.loss_plot('epoch', result_path)
    del model, history
    return res


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
