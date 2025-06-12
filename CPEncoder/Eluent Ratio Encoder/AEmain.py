import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
# from torchvision import transforms,datasets
from torch import nn,optim

from AutoEncoder import AE

# import visdom
pro_path = '../../'

def train_AE():
    # 读取数据集
    train_data_path = os.path.join(pro_path, 'data', 'processed', 'AEvec', 'eluent_vec.csv')
    train_data_df = pd.read_csv(train_data_path, sep=',')
    # 将nan填充为0
    train_data_df = train_data_df.fillna(0)
    print(f"训练数据总大小：{train_data_df.shape}")
    # 每一列都是特征 共54维
    data = train_data_df.values
    # 转为浮点型张量
    data_tensor = torch.tensor(data, dtype=torch.float32)
    print("data_tensor_type: ", type(data_tensor))
    # 创建 TensorDataset (数据集) 无标签
    dataset = TensorDataset(data_tensor)
    print("dataset_type: ",type(dataset))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #实例化自编码器
    model = AE(device)#.to(device)
    # model = model.to(device)
    #评价标准
    criterion = nn.MSELoss()
    #优化器
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    print('model:\n',model)
    # 定义要搜索的 batch_size 和 epoch 的范围
    epochs_list = [10, 20, 40, 80]  # 不同训练epochs
    batch_size_list = [16, 32, 64, 128]  # 每次送入的数据量

    best_loss = float('inf')  # 初始化为一个非常大的值
    best_model_wts = None  # 用于保存最佳模型权重
    best_params = {}  # 用于保存最佳的超参数组合

    for epochs in epochs_list:
        for batch_size in batch_size_list:
            # 生成批次数据
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            # 迭代epoch
            for epoch in range(epochs):
                cur_loss = 0.0
                for batch_data in dataloader:
                    # 提取输入 无监督学习没有标签 因此batch_data是只包含一个tensor的列表
                    x = batch_data[0]

                    x.to(device)
                    x_hat = model(x, device)
                    # 出现nan
                    if torch.isnan(x).any():
                        print("出现nan")
                        exit(0)
                        # break
                    '''
                    x1 = model.encoder[0](x)
                    print("第一层: ", x1)
                    x2 = model.encoder[1](x1)
                    print("第二层: ", x2)
                    '''

                    loss = criterion(x_hat,x)

                    # 反向传播
                    # 清空前面批次的梯度
                    optimizer.zero_grad()
                    # 反向传播 计算梯度
                    loss.backward()
                    # 根据梯度使用优化器更新模型参数
                    optimizer.step()

                    cur_loss += loss.item()
                    # print(loss.item())
                # print(epoch,'loss:',loss.item())
                epoch_loss = cur_loss / len(dataloader)
                print(f'Epoch [{epoch + 1}/{epochs}], Batch Size: {batch_size}, Loss: {epoch_loss}')

                # 检查是否是最小的 loss
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()  # 保存当前模型的权重
                    best_params = {'epochs': epochs, 'batch_size': batch_size}  # 记录最佳超参数
        pass
    print(f"best_epochs:{best_params['epochs']}, best_batch_size:{best_params['batch_size']}")
    # 保存最佳模型
    os.makedirs('./model', exist_ok=True)
    torch.save(best_model_wts, './model/best_model.pth')
    print(f'Best Model found with Loss: {best_loss}, Params: {best_params}')
    print('-'*50)

def getVector():
    # gradient & eluent data
    # data_path = os.path.join(pro_path, 'data', 'processed', 'AEvec', 'eluent')
    data_path = os.path.join(pro_path, 'data', 'origin', 'original_eluent')
    result_path = os.path.join(pro_path, 'data', 'processed', 'AEvec', 'AEvectors', 'eluent')
    os.makedirs(result_path, exist_ok=True)
    file_list = os.listdir(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = AE(device)
    # 加载模型参数
    autoencoder.load_state_dict(torch.load('./model/best_model.pth'))
    # 设置模型为评估模式：禁用 Dropout 和 BatchNorm 等训练期间特定的行为
    autoencoder.eval()

    # print(file_list)
    # 使用模型进行推理

    for file in file_list:
        print(f"提取{file}表示")
        # 文件路径
        file_path = os.path.join(data_path, file)
        df = pd.read_csv(file_path, sep=',')
        x = torch.tensor(df.values, dtype=torch.float32)
        number = file.split('_')[0].split('.')[0]
        file_result_path = os.path.join(result_path, f"{number}_AE_eluent_vector.csv")
        with torch.no_grad():  # 禁用梯度计算（因为我们只是在推理）
            # 提取编码器表示
            representation = autoencoder.encoder(x)
            # 对所有行求和
            # 洗脱液仅一行 求和可省略
            encoded_representation = torch.sum(representation, dim=0)
            # encoded_representation = representation
            # print('-'*20)
            # print(encoded_representation)

            if encoded_representation.is_cuda:
                encoded_representation = encoded_representation.cpu()
            # 转为一列numpy数组
            numpy_array = encoded_representation.numpy()
            # print(numpy_array)
            # numpy数组转为一行df
            df = pd.DataFrame([numpy_array])
            # print(df)
            # 写出csv
            df.to_csv(file_result_path, index=False,header=None)
            # print('-' * 20)
        pass
    pass




if __name__ == '__main__':

    start_time = time.time()


    # 训练
    # train_AE()
    # 提取
    getVector()

    end_time = time.time()

    # 运行时间计算
    elapsed_time = end_time - start_time

    days = int(elapsed_time // (24 * 3600))
    elapsed_time = elapsed_time % (24 * 3600)
    hours = int(elapsed_time // 3600)
    elapsed_time %= 3600
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print(f"总用时：{days}天 {hours}小时 {minutes}分 {seconds}秒")