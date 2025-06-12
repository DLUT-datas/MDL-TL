import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model


# 负责创建模型
def build_DNN(num_lr, model_path):
    np.random.seed(1)
    tf.random.set_seed(1)

    model = load_model(model_path + "base_model.h5")

    adam = keras.optimizers.Adam(lr=num_lr)
    model.compile(optimizer=adam, loss='mae')
    return model
