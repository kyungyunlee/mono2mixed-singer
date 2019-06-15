import os 
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Dense, Dropout, Activation, Reshape, Input, Concatenate, dot, Add, Multiply, Flatten, concatenate, LeakyReLU, Lambda, Merge
from keras.models import Model
from keras.regularizers import l2





def basic_cnn(num_frame, num_artist):
    x_input = Input(shape=(num_frame, 128))
    
    # axis1=freq, axis2=time, axis3=channel
    # axis1=freq, axis2=time
    # out = BatchNormalization(axis=1)(x_input)

    out = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x_input)
    out = BatchNormalization(axis=2)(out)
    out = LeakyReLU(0.2)(out)
    out = MaxPool1D(pool_size=3)(out)

    out =Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(out)
    out = BatchNormalization(axis=2)(out)
    out = LeakyReLU(0.2)(out)
    out = MaxPool1D(pool_size=3)(out)

    out = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(out)
    out = BatchNormalization(axis=2)(out)
    out = LeakyReLU(0.2)(out)
    out = MaxPool1D(pool_size=3)(out)
    out = Dropout(0.5)(out)
    
    out = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(out)
    out = BatchNormalization(axis=2)(out)
    out = LeakyReLU(0.2)(out)
    out = MaxPool1D(pool_size=3)(out)
    
    out = Conv1D(256, kernel_size=1, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(out)
    out = BatchNormalization(axis=2)(out)
    out = LeakyReLU(0.2)(out)
    out = Dropout(0.5)(out)

    
    out = GlobalAvgPool1D()(out)


    out = Dense(num_artist, activation='softmax')(out)
    model = Model(inputs=x_input, outputs = out)
    return model



if __name__ == '__main__':
    model = basic_2d_cnn(129, 1000)
    print (model.summary())
