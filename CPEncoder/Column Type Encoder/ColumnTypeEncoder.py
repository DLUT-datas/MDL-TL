import os
import csv

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

def getColumns(column_database_path):
    '''
    读取色谱柱信息
    :param column_database_path: 色谱柱数据库路径（来自RepoRT）
    :return:
    '''
    # 读取所有色谱柱信息
    column_database = pd.read_csv(column_database_path, sep='\t', header=0, index_col=None,
                                  usecols=[r'column'], encoding='unicode_escape')
    print(column_database.shape)
    # 去重
    # column_database.drop_duplicates()
    columns = column_database.iloc[:, 0].tolist()
    columns = list(set(columns))
    return columns

def train_model(document, model_path, model_file):


    # Word2Vec参数设置，来自Spec2Vec
    args = {
        "sg": 0,
        "negative": 5,
        "vector_size": 10,
        "window": 500,
        "min_count": 1,
        "alpha": 0.025,
        "min_alpha": 0.00025,
        "workers": 4,
        "compute_loss": True,
    }
    # 构建模型
    model = Word2Vec(document, **args)
    model_full_path = os.path.join(model_path, model_file)
    # 保存模型文件
    model.save(model_full_path)


def getColumnVec(model_path, model_file, columns_path, vec_save_path):
    '''
    为所有色谱柱类型提取向量表示（单独存储）
    :return: 
    '''
    model = Word2Vec.load(os.path.join(model_path, model_file))


    files = os.listdir(columns_path)
    files = [file for file in files if file.endswith('.txt')]
    for file in files:
        path = os.path.join(columns_path, file)
        words = []
        column_vec = np.zeros(model.vector_size)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.readline().strip()
            words = content.split()
        for word in words :
            if word in words:
                column_vec += model.wv[word]
            else:
                print(f"{words} not in meta.model")

        vec_file_save_path = os.path.join(vec_save_path, file[:-4]+'.csv')
        if not os.path.exists(vec_save_path):
            os.makedirs(vec_save_path)

        with open(vec_file_save_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_vec.tolist())
    pass
def main():
    # column_database.tsv路径
    column_database_path = r'.\workspace\RepoRT-master\resources\column_database\column_database.tsv' # 修改为自己的本地路径
    # 数据集要使用的色谱柱类型
    columns_path = '../../data/origin/original_columns'  # 将要使用的色谱柱类型放入txt文件中（单个文件存储单个类型）
    # 向量存储地址
    vec_save_path = '../../data/processed/column_vector'
    os.makedirs(vec_save_path, exist_ok=True)
    # model路径
    model_path = r'./model'
    os.makedirs('./model', exist_ok=True)
    # 模型文件名称
    model_file = 'meta.model'
    # 获取柱类型
    # columns = getColumns(column_database_path)
    # 将柱类型的字符串转换为词列表（以空格为分隔符） 每个柱的字符串是一个句子
    # sentences = [s.split() for s in columns]
    # print(sentences)
    # 构建模型
    # train_model(sentences, model_path, model_file)
    # 提取表示
    getColumnVec(model_path, model_file, columns_path, vec_save_path)



main()

