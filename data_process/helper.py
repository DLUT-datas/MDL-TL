# coding=utf-8
import numpy as np
import csv
import pandas as pd
from mordred import Calculator, descriptors
import math
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys


# csv文件的读取
def read_csv(filename, header=0):
    df = pd.read_csv(filename, header=header)  # 默认第一行为表头，自动生成索引, dataFrame类型
    a = df.values  # a是numpy类型
    return a


# csv文件的写入
def write_csv(filename, rows):
    # 写入csv文件: 将list写入csv文件
    # headers = ['class', 'name', 'sex', 'height', 'year']
    # rows = [
    #     [1, 'xiaoming', 'male', 168, 23],
    #     [1, 'xiaohong', 'female', 162, 22],
    #     [2, 'xiaozhang', 'female', 163, 21],
    #     [2, 'xiaoli', 'male', 158, 21]
    # ]
    with open(filename, 'w', newline="") as f:  # newline="" 是为了去掉行与行之间的空格
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
    smiles, rts = [], []
    with open(filename, 'r') as f:
        data = f.readlines()[1:]  # 过滤掉表头
        for line in data:
            a, b = line[:-1].split('\t')
            smiles.append(a)
            rts.append(float(b))
    # print(data)  # data是list类型
    return smiles, rts


# smiles, rts = read_txt(pro_path + "data/smiles_rt/Aicheler.txt")
# print(smiles)
# print(rts)

####################################################################################
import sys
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
import preprocess as pp
import pickle

radius = 1
dim = 48
layer_hidden = 6
layer_output = 6
batch_train = 32
batch_test = 32
lr = 1e-4
lr_decay = 0.85
decay_interval = 10
iteration = 200
N = 5000

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')



class Tester(object):
    def __init__(self, model):
        self.model = model

    def test_regressor(self, dataset, rts, gen=False):  # 提取训练得到的向量
        N = len(dataset)
        SMILES, Ts, Ys = '', [], []
        data = []
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i + batch_test]))
            (Smiles, molecular_vectors) = self.model.forward_regressor(data_batch, train=False)
            SMILES += ' '.join(Smiles) + ' '
            correct_values = rts[i:i + batch_test]

            if gen:  # 添加
                for j in range(len(Smiles)):
                    item = [Smiles[j], correct_values[j]]
                    for k in range(len(molecular_vectors[j])):
                        item.append(molecular_vectors[j][k].item())
                    data.append(item)
        return np.array(data)




####################################################################################

def cal_MACCS(new_smiles, new_rts, mols):
    data = []
    cnt = 0  # 进度
    for i in range(len(mols)):
        mol = mols[i]
        cnt += 1
        if cnt % 100 == 0:
            print(cnt, "/", len(mols))

        # 计算指纹
        fps = MACCSkeys.GenMACCSKeys(mol)
        # 指纹转化为数组
        fp_arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fps, fp_arr)
        # 数组中每个元素转化为int, 因为只可能是0或者1
        fp_arr = [int(e) for e in fp_arr]  # list

        data.append([new_smiles[i], new_rts[i]] + fp_arr[1:])  # 第0位是占位符，需要删除
    return np.array(data)


####################################################################################

def cal_mordred(new_smiles, new_rts, mols):
    calc = Calculator(descriptors, ignore_3D=True)
    data = []
    cnt = 0  # 用于查看进度
    for i in range(len(mols)):
        mol = mols[i]
        item = []

        res = calc(mol)
        cnt += 1
        if cnt % 20 == 0:
            print(cnt, "/", len(mols))

        for e in res.values():
            e = float(e)
            if math.isnan(e):  # 是NaN的话填充字符串：loss
                item.append("loss")
            else:
                item.append(e)
        data.append([new_smiles[i], new_rts[i]] + item)
    return np.array(data)


def fill_mordred(data, mols):
    fps = [Chem.RDKFingerprint(x) for x in mols]
    mp = {}
    topK = 5
    new_data = [data[:, 0], data[:, 1]]
    data = data[:, 2:].T
    cnt = 0
    for item in data:  # 填充每个特征
        index_list = []
        loss_list = []

        cnt += 1
        if cnt % 100 == 0:
            print(cnt, "/", len(data))
            print("len(mp):", len(mp))

        for i in range(len(item)):
            if item[i] == "loss":
                loss_list.append(i)
            else:
                index_list.append(i)
        if len(loss_list) == 0:
            new_data.append(item)
            continue

        for i in loss_list:
            rank_list = []
            for j in index_list:
                tp = (i, j)
                if tp in mp:
                    rank_list.append(mp[tp])
                else:
                    sim = DataStructs.FingerprintSimilarity(fps[i], fps[j])
                    rank_list.append((sim, j))
                    if len(mp) < 2e7:
                        mp[(i, j)] = (sim, j)

            rank_list.sort(key=lambda x: -x[0])
            t = 0
            for j in range(min(len(rank_list), topK)):
                t += float(item[rank_list[j][1]])
            item[i] = t / max(min(len(rank_list), topK), 1)  # rank_list可能为空，说明全部是缺失值
            del rank_list
        new_data.append(item)

    new_data = np.array(new_data).T
    return new_data


####################################################################################

def standred(data):
    data = data.T
    new_data = [data[0], data[1]]

    for i in range(2, len(data)):
        item = data[i]
        item = np.array([float(e) for e in item])
        mean = np.mean(item)
        std = np.std(item)
        if std != 0:
            new_item = (item - mean) / std
            new_data.append(new_item)
        else:
            new_data.append(item)

    new_data = np.array(new_data).T
    return new_data


