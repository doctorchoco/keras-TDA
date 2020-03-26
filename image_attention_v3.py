from keras.engine import Model
from keras_vggface.vggface import VGGFace
from keras.layers import Dropout, Flatten, Dense, Activation, Convolution2D, MaxPooling2D, PReLU, Input, Activation, \
    add, Dense, Flatten, Dropout, Multiply, Embedding, Lambda, Conv2D
from keras.layers import *
from keras import backend as K
from keras.layers import Bidirectional
from keras import activations
import tensorflow as tf
import keras
### Parameter setting

import os

from keras.initializers import Constant, RandomNormal, he_normal
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def base_model_v1():

    base_input = Input(shape=(48,48,3), name='b_input_1')
    model = Conv2D(16, (3, 3), padding='valid', name='b_conv_1', activation='tanh')(base_input)
    model = MaxPooling2D(pool_size=2, strides=2, padding='valid', name='b_max_1')(model)

    model = Conv2D(32, (3, 3), padding='valid', name='b_conv_2', activation='tanh')(model)
    model = MaxPooling2D(pool_size=2, strides=2, padding='valid', name='b_max_2')(model)

    model = Conv2D(64, (3, 3), padding='valid', name='b_conv_3', activation='tanh')(model)
    model = MaxPooling2D(pool_size=2, strides=2, padding='valid', name='b_max_3')(model)

    model = Conv2D(32, (1, 1), padding='valid', name='b_conv_4', activation='tanh')(model)
    model = MaxPooling2D(pool_size=4, strides=4, padding='valid', name='b_max_4')(model)

    model = Flatten(name='flatten')(model)

    model = Dense(64, activation='relu', name='b_dense_1')(model)
    #    model = Dropout(0.25)(model)
    model = Dense(16, activation='relu', name='b_dense_2')(model)
    #    model = Dropout(0.25)(model)
    model = Dense(5, name= 'b_dense_3')(model)
#    sum = Lambda(lambda x: K.sum(x), name='b_sum')(model)
    base_model = Model(base_input, model, name="base_model")
#    base_model.summary()
    return base_model


def attention_model_v1(channel_size):

    input_layer = Input(shape=(5,), name='a_input')
    atten_model = input_layer
    atten_model = Dense(16, name="a_dense_1", activation='relu')(atten_model)
    atten_model = Dense(32, name="a_dense_2", activation='relu')(atten_model)
    atten_model = Dense(channel_size, name="a_dense_3", kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=42), bias_initializer=RandomNormal(mean=0, stddev=0.001, seed=42))(atten_model)
#    atten_model = Activation('sigmoid', name='a_sigmoid')(atten_model)
#    atten_model = Activation('linear', name='a_sigmoid')(atten_model)
    atten_model = Activation('relu', name='a_sigmoid')(atten_model)
    atten_model = Lambda(lambda x: x + 1, name='a_multiply')(atten_model)
#    atten_model = Lambda(lambda x: x + 0.5, name='atten_3')(atten_model)
    model = Model(input_layer, atten_model, name="attention_model")
#    model.summary()
#    model.save(model_name)
    return model

def integration_model_v1(iteration, al):
    rec_num = iteration
    base_model=base_model_v1()
    base_atten = base_model.layers[al]

    channel_size = base_atten.output_shape[-1]
    attnetion_model=attention_model_v1(channel_size)

    integ_out = base_model.output

    for i in range(0, rec_num):
        atten_out = attnetion_model(base_model.output)
        next_layer = multiply([base_atten.output, atten_out])
        integ_out = next_layer

    integ_out = Activation('softmax')(integ_out)
    model = Model(base_model.input, integ_out, name="integraion_model")
    model.summary()

    return model
