# coding=utf-8
import os
import csv
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 项目所在根目录
pro_path = "../"


# csv文件的读取
def read_csv(filename, header=0):
    df = pd.read_csv(filename, header=header)  # 默认第一行为表头，自动生成索引, dataFrame类型
    a = df.values  # a是numpy类型
    return a


# csv文件的写入
def write_csv(path, filename, rows):
    # 写入csv文件: 将list写入csv文件
    # headers = ['class', 'name', 'sex', 'height', 'year']
    # rows = [
    #     [1, 'xiaoming', 'male', 168, 23],
    #     [1, 'xiaohong', 'female', 162, 22],
    #     [2, 'xiaozhang', 'female', 163, 21],
    #     [2, 'xiaoli', 'male', 158, 21]
    # ]
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, filename), 'w', encoding='utf-8', newline="") as f:  # newline="" 是为了去掉行与行之间的空格
        writer = csv.writer(f)
        writer.writerows(rows)


# 写入txt文件
def write_txt(filename, con):
    # 写入文件
    # con = ["hello\n", "12.0\n"]
    with open(filename, 'w') as f:
        f.writelines(con)


# 文本文件的处理: 每行数据是纯数字
def read_txt(filename):
    # 读取文件
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(line[:-1])
    # print(data)  # data是list类型
    return data


# 文本文件的处理: 每行数据是纯数字
def read_txt_smiles_rts(filename):
    # 读取文件
    smiles, rts = [], []
    with open(filename, 'r') as f:
        data = f.readlines()[1:]  # 过滤掉表头
        for line in data:
            a, b = line[:-1].split('\t')
            smiles.append(a)
            rts.append(float(b))
    # print(data)  # data是list类型
    return smiles, rts


def cal_relative_median_error(val, pre):
    res = []
    for i in range(val.shape[0]):
        res.append(np.abs((val[i] - pre[i]) / val[i]))
    return res


def cal_r_square(y_origin, y_predict):
    return r2_score(y_origin, y_predict)


def cal_relative_error(y_origin, y_predict):
    res = []
    for i in range(len(y_origin)):
        res.append(np.abs((y_origin[i] - y_predict[i]) / y_origin[i]))
    return res


def cal_index(y_origin, y_predict):
    error_val = list(map(lambda o, p: o - p, y_origin, y_predict))
    error_val_abs = list(map(abs, error_val))

    # ------ 计算绝对误差 ------
    # 计算平均误差
    error_mean = np.mean(error_val_abs)  # MAE
    # 计算中位数误差
    error_median = np.median(error_val_abs)  # MedAE

    # ------ 计算相对误差 ------
    relative_error_abs = cal_relative_error(y_origin, y_predict)
    # 计算相对中位数平均误差
    relative_error_mean = np.mean(relative_error_abs)  # MRE
    # 计算相对中位数误差
    relative_error_median = np.median(relative_error_abs)  # MedRE

    # ------ 计算R^2 ------
    R_square = cal_r_square(y_origin, y_predict)

    # ------ 计算RMSE(均方根误差) ------
    RMSE = np.sqrt(np.mean(np.square(y_origin - y_predict)))

    res = [
        error_mean,  # MAE
        error_median,  # MedAE
        relative_error_mean,  # MRE
        relative_error_median,  # MedRE
        R_square,
        RMSE,
    ]
    result = []
    for i in range(len(res)):
        result.append(round(res[i], 4))

    return result


def cost_time(start, end):
    return str(round(end - start, 1) // 60) + " min " + str(round(end - start, 0) % 60) + "s"


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
        write_csv(path, "loss.csv", np.array(self.losses[loss_type]).reshape(1, -1))
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
