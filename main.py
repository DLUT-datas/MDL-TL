# coding=utf-8
import os
import numpy as np
import pandas as pd

import sys
import utils


data_path = "./data/"
pro_path = "./"


for i in range(10):
    seed = i


    # 删除重复元素，删除rt=0的数据，新文件写入data/target/中
    def file_process(X, file, save_filename):
        S = set()
        dup = []
        data = []
        for i in range(len(X)):
            smile, rt = X[i][0], X[i][1]
            if smile in S:  # 重复保留
                dup.append(smile)
            else:
                if rt == 0:
                    print(file + ": " + smile + ", RT=0")
                    continue
            data.append(list(X[i]))
            S.add(smile)
        print(file + " dup num: " + str(len(dup)))
        utils.write_csv(data_path + "target/" + file + "/", save_filename, data)


    # 读取小数据集中的表示，删除重复元素，删除rt=0的数据，新文件写入data/target/中
    def f1():
        file_list = os.listdir(data_path + "smiles_rt/")

        input_dirs = []
        for file in file_list:
            if len(file.split('.')) == 2:  # 说明是文件
                continue
            input_dirs.append(file)


        print(input_dirs)
        print(len(input_dirs))

        for i in range(len(input_dirs)):
            file = input_dirs[i]

            path = data_path + "smiles_rt/" + file + "/"
            print(path)
            # 读取数据
            X1 = utils.read_csv(path + file + "_maccs.csv", header=None)
            X3 = utils.read_csv(path + file + "_mordred_std.csv", header=None)

            # 处理数据
            file_process(X1, file, file + "_maccs.csv")
            file_process(X3, file, file + "_mordred_std.csv")

        pass


    f1()


    # 将n个数据划分，下标存入文件
    def my_split(n, seed):
        # n : 样本数量，划分为train、val、test，比例是81:9:10
        # seed : 随机种子
        train_val_num = int(n * 0.9)
        test_num = n - train_val_num  # 测试集数量
        train_num = int(train_val_num * 0.9)  # 训练集数量
        val_num = train_val_num - train_num  # 验证集数量

        l = [i for i in range(n)]
        np.random.seed(seed)
        np.random.shuffle(l)

        train_index, val_index, test_index = l[0:train_num], l[train_num:train_val_num], l[train_val_num:]

        return train_index, val_index, test_index


    # 划分数据集为train、val、test，比例是81:9:10
    def f2():
        file_list = os.listdir(data_path + "target/")

        input_dirs = []
        for file in file_list:
            if len(file.split('.')) == 2:  # 说明是文件
                continue
            input_dirs.append(file)

        print(input_dirs)
        print(len(input_dirs))

        for i in range(len(input_dirs)):
            file = input_dirs[i]

            path = data_path + "target/" + file + "/"
            print(path)
            # # 读取数据
            X1 = utils.read_csv(path + file + "_maccs.csv", header=None)
            # X3 = utils.read_csv(path + file + "_mordred_std.csv", header=None)

            train_index, val_index, test_index = my_split(len(X1), seed=seed)
            print(len(train_index), len(val_index), len(test_index))
            utils.write_csv(path + "split_index/np_seed" + str(seed) + "/", "train_index.csv", [train_index])
            utils.write_csv(path + "split_index/np_seed" + str(seed) + "/", "val_index.csv", [val_index])
            utils.write_csv(path + "split_index/np_seed" + str(seed) + "/", "test_index.csv", [test_index])


    f2()


    # 根据target/文件夹的数据，生成训练集、验证集、测试集
    # 训练集：将所有训练集数据混合在一起，每个数据集需要加上编码后的色谱参数向量为特征。
    # 验证集：将所有验证集数据混合在一起，用来验证
    # 测试集：测试集数据分开，用于得到各个数据集的测试结果
    def f3():
        file_list = os.listdir(data_path + "target/")

        input_dirs = []
        for file in file_list:
            if len(file.split('.')) == 2:  # 说明是文件
                continue
            input_dirs.append(file)


        print(input_dirs)
        print(len(input_dirs))
        column_prop_path = os.path.join(data_path, 'column_head.csv')
        column_prop = pd.read_csv(column_prop_path, header=None).iloc[:, 0].tolist()
        X_train, X_val, X_test = [], [], []  # X_train, X_val是二维数据，X_test是三维数据(因为没有打乱)
        y_train, y_val, y_test = [], [], []  # y_train, y_val是一维数据，y_test是二维数据(因为没有打乱)
        for i in range(len(input_dirs)):
            file = input_dirs[i]

            path = data_path + "target/" + file + "/"
            print(path)
            # 读取数据

            X1 = utils.read_csv(path + file + "_maccs.csv", header=None)

            X3 = utils.read_csv(path + file + "_mordred_std.csv", header=None)
            y = X1[:, 1]
            X = np.concatenate([X1[:, 2:], X3[:, 2:]], axis=1)


            X, y = X.astype("float64"), y.astype("float64")

            # '''
            #  ********************************** 拼接 色谱参数 向量 **********************************
            #  ---------------------------------- 拼接 梯度 & 洗脱液 向量 ----------------------------------
            # 梯度+洗脱液编码 各32维 共64维

            # 使用AE编码后的梯度与洗脱液向量 data/processed/AEvec/AEvectors/0001_AEvector.csv
            vector_path = os.path.join(data_path, "processed/AEvec/AEvectors/")
            gradient_path = os.path.join(vector_path, "gradient")
            eluent_path = os.path.join(vector_path, "eluent")
            filename = file.split('_')[0]

            gradient_vector_file = os.path.join(gradient_path, f"{filename}_AE_gradient_vector.csv")
            eluent_vector_file = os.path.join(eluent_path, f"{filename}_AE_eluent_vector.csv")
            # 读取对应编码向量
            gradient_vector_df = pd.read_csv(gradient_vector_file, header=None)
            eluent_vector_df = pd.read_csv(eluent_vector_file, header=None)
            # 拼接梯度与洗脱液向量
            gradient_eluent_vector = pd.concat([gradient_vector_df, eluent_vector_df], axis=1)
            print(f"梯度与洗脱液向量拼接: {gradient_eluent_vector.shape}")
            # 获取第一行（向量文件只有一行）
            gradient_eluent_vector = gradient_eluent_vector.values[0]


            # 给同一数据集下的所有向量末尾都拼接同一个梯度+洗脱液向量
            gradient_eluent_vector_repeated = np.tile(gradient_eluent_vector, (X.shape[0], 1))
            X = np.hstack((X, gradient_eluent_vector_repeated))
            #  ---------------------------------- 拼接 梯度 & 洗脱液 向量 ----------------------------------

            # '''

            # '''
            #  ---------------------------------- 拼接 数值型参数 向量 ----------------------------------
            # 数值型色谱参数 6维
            # 将色谱条件加入输入特征中 洗脱液成分、柱属性等
            meta_vector_path = os.path.join(data_path, "processed/meta_vector/")
            meta_filename = file.split('_')[0] + '_metainfo.csv'
            meta_vector_file = os.path.join(meta_vector_path, meta_filename)
            # 读取对应色谱条件向量 带有表头 header不为空
            meta_vector_df = pd.read_csv(meta_vector_file, usecols=column_prop)
            print("色谱数值型参数：")
            print(meta_vector_df)
            # 将死时间转换为秒
            meta_vector_df['column.t0'] = meta_vector_df['column.t0'] * 60

            meta_vector_from_csv = meta_vector_df.values[0]
            # 给同一环境下的所有向量末尾都拼接同一个色谱条件向量
            meta_vector_repeated = np.tile(meta_vector_from_csv, (X.shape[0], 1))
            X = np.hstack((X, meta_vector_repeated))
            # print(X)
            # '''
            #  ---------------------------------- 拼接 数值型参数 向量 ----------------------------------


            # '''
            #  ---------------------------------- 拼接 色谱柱类型 向量 ----------------------------------

            # 色谱柱类型10维
            # 将编码后的色谱柱类型加入特征中
            column_vector_path = os.path.join(data_path, "processed/column_vector/")
            column_filename = file.split('_')[0] + '_column.csv'
            column_vector_file = os.path.join(column_vector_path, column_filename)
            # 读取对应色谱柱向量
            column_vector_df = pd.read_csv(column_vector_file, header=None)
            column_vector_from_csv = column_vector_df.values[0]
            # 给同一环境下的所有向量末尾都拼接同一个色谱柱向量
            column_vector_repeated = np.tile(column_vector_from_csv, (X.shape[0], 1))
            # 共 1427维
            X = np.hstack((X, column_vector_repeated))
            # '''
            #  ---------------------------------- 拼接 色谱柱类型 向量 ----------------------------------

            #  ********************************** 拼接 色谱参数 向量 **********************************

            print(X.shape)




            # 从文件中读取训练集下标、验证集下标、训练集下标
            train_index = \
                utils.read_csv(path + "split_index/np_seed" + str(seed) + "/" + "train_index.csv", header=None)[0]
            val_index = utils.read_csv(path + "split_index/np_seed" + str(seed) + "/" + "val_index.csv", header=None)[0]
            test_index = utils.read_csv(path + "split_index/np_seed" + str(seed) + "/" + "test_index.csv", header=None)[
                0]

            # 根据下标得到真实数据
            X_train_tmp, X_val_tmp, X_test_tmp = X[train_index], X[val_index], X[test_index]
            y_train_tmp, y_val_tmp, y_test_tmp = y[train_index], y[val_index], y[test_index]


            # 单独数据写入文件
            utils.write_csv(data_path + "gen/np_seed" + str(seed) + "/" + file + "/", "X_train.csv", X_train_tmp)
            utils.write_csv(data_path + "gen/np_seed" + str(seed) + "/" + file + "/", "y_train.csv", [y_train_tmp])
            utils.write_csv(data_path + "gen/np_seed" + str(seed) + "/" + file + "/", "X_val.csv", X_val_tmp)
            utils.write_csv(data_path + "gen/np_seed" + str(seed) + "/" + file + "/", "y_val.csv", [y_val_tmp])
            utils.write_csv(data_path + "gen/np_seed" + str(seed) + "/" + file + "/", "X_test.csv", X_test_tmp)
            utils.write_csv(data_path + "gen/np_seed" + str(seed) + "/" + file + "/", "y_test.csv", [y_test_tmp])

            # 汇总训练集数据
            num = len(X_train_tmp)
            for u in range(num):
                X_train.append(X_train_tmp[u])
            for u in range(num):
                y_train.append(y_train_tmp[u])

            # 汇总验证集数据
            num = len(X_val_tmp)
            for u in range(num):
                X_val.append(X_val_tmp[u])
            for u in range(num):
                y_val.append(y_val_tmp[u])

            # 测试集数据（不用汇总）
            # X_test.append(X_test_tmp)
            # y_test.append(y_test_tmp)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        # 打乱训练集顺序
        index = [i for i in range(len(X_train))]
        np.random.shuffle(index)
        X_train, y_train = X_train[index], y_train[index]

        # 打乱验证集顺序
        index = [i for i in range(len(X_val))]
        np.random.shuffle(index)
        X_val, y_val = X_val[index], y_val[index]

        print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

        # 将训练集、验证集写入csv文件
        utils.write_csv(data_path + "gen/np_seed" + str(seed) + "/", "X_train.csv", X_train)
        utils.write_csv(data_path + "gen/np_seed" + str(seed) + "/", "y_train.csv", [y_train])
        utils.write_csv(data_path + "gen/np_seed" + str(seed) + "/", "X_val.csv", X_val)
        utils.write_csv(data_path + "gen/np_seed" + str(seed) + "/", "y_val.csv", [y_val])


    f3()



