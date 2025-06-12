import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras import initializers
from keras.models import Model


# 负责创建模型
def build_DNN(input_shape, num_lr, init_val):
    np.random.seed(1)
    tf.random.set_seed(1)

    layer_init_size = pow(10, len(str(input_shape))-1)

    init = None
    if init_val == "random_normal":
        init = initializers.random_normal(mean=0.0, stddev=0.05, seed=None)
    elif init_val == "glorot_normal":
        init = initializers.glorot_normal(seed=1)
    elif init_val == "glorot_uniform":
        init = initializers.glorot_uniform(seed=1)

    input_md = Input(shape=(input_shape,), name='input_md')
    x_md1 = Dense(int(layer_init_size * 1), kernel_initializer=init, bias_initializer='zeros', activation='relu', name='AE1')(input_md)
    x_md2 = Dense(int(layer_init_size * 0.8), kernel_initializer=init, bias_initializer='zeros', activation='relu', name='AE2')(x_md1)
    x_md3 = Dense(int(layer_init_size * 0.4), kernel_initializer=init, bias_initializer='zeros', activation='relu', name='AE3')(x_md2)
    x_md4 = Dense(int(layer_init_size * 0.2), kernel_initializer=init, bias_initializer='zeros', activation='relu', name='AE4')(x_md3)
    x_md5 = Dense(int(layer_init_size * 0.1), kernel_initializer=init, bias_initializer='zeros', activation='relu', name='AE5')(x_md4)
    output = Dense(1, kernel_initializer=init, bias_initializer='zeros', name='output')(x_md5)

    model = Model([input_md], [output])
    adam = keras.optimizers.Adam(lr=num_lr)
    model.compile(optimizer=adam, loss='mae')
    return model
