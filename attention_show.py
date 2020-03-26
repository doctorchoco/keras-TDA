from keras.layers import Dropout, Flatten, Dense, Activation, Convolution2D, MaxPooling2D, PReLU, Input, Activation, \
    add, Dense, Flatten, Dropout, Multiply, Embedding, Lambda, Conv2D

import tensorflow as tf

from keras.engine import Model
import numpy as np
import glob
import os
from keras import backend as K

from keras import models

import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array, load_img

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


logdir = "/data3/younghoon/Flagship_2019/Multimodal_2019/image_5emo/data/1217_1607/"


weight = "weights.112-0.99.h5"

model = models.load_model(logdir + weight)

model_base = Model(inputs=model.input, outputs=model.get_layer(name="b_dense_3").get_output_at(0))

model_1 = Model(inputs=model.input, outputs=model.get_layer(name="b_max_3").get_output_at(0))

model_2 = Model(inputs=model.get_layer(name="attention_model").get_layer(name="a_input").get_output_at(0),
                outputs=model.get_layer(name="attention_model").get_layer(name="a_multiply").output)

Inputs_1 = Input(shape=(4, 4, 64,))
Inputs_2 = Input(shape=(64,))
hello = model.get_layer(name="multiply_1")([Inputs_1, Inputs_2])
hello = model.get_layer(name="b_conv_4")(hello)
hello = model.get_layer(name="b_max_4")(hello)
hello = model.get_layer(name="flatten")(hello)
hello = model.get_layer(name="b_dense_1")(hello)
hello = model.get_layer(name="b_dense_2")(hello)
hello = model.get_layer(name="b_dense_3")(hello)
hello = model.get_layer(name="activation_1")(hello)
model_3 = Model(inputs=[Inputs_1, Inputs_2], outputs=hello)

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def attention_out(data_dir):
    img = img_to_array(load_img(data_dir))
    img = img.reshape(1, 48, 48, 3)
    img=img.astype('float32') / 255

    dense_3 = model_base.predict(img)
    base_pred = dense_3[0]
    base_pred = softmax(base_pred)
    base_pred = [round(x, 2) for x in base_pred]
    print('base result:', np.argmax(base_pred) , base_pred)

    att_pred = model.predict(img)
    att_pred = att_pred[0]
    att_pred = [round(x, 2) for x in att_pred]
    print('Attention_result:', np.argmax(att_pred), att_pred)


    feat_1=model_1.predict(img)
    dim=len(np.shape(feat_1))
    for i in range(0,dim-1):
        feat_1_a = np.average(feat_1,0)
#    print(feat_1_a)

    feat_2=model_2.predict(dense_3)
    feat_2_a=feat_2[0]
    attention_out = [round(x,2) for x in feat_2_a]

    return attention_out


def attention_out_modi(data_dir, atten_num):
    img=img_to_array(load_img(data_dir))
    img=img.reshape(1,48,48,3)
#    input_img = np.zeros((1, 48, 48, 3))
#    input_img[0] = img
    img=img.astype('float32') / 255
    #print(model.get_layer(name="base_model").get_config())

#    model_1= Model(inputs=model.get_layer(name="base_model").get_layer(name="input_2").input, outputs=model.get_layer(name="base_model").get_layer(name="b_dense_3").output)
#    model_1= Model(inputs=model.get_layer(name="b_base_layer").get_layer(name="input_1").input, outputs=model.get_layer(name="b_base_layer").get_layer(name="b_dense_3").output)

#    feat_1=model_1.predict(feat_0)

#    model_0 = Model(inputs=model.input, outputs=model.get_layer(name="b_dense_3").get_output_at(0))
    feat_0 = model_base.predict(img)
    feat_0 = feat_0[0]
    result_1 = np.argmax(feat_0)
#    result_1 = [round(x, 2) for x in result_1]
    print('base result:', result_1)

    result=model.predict(img)
    print(result)
    result=result[0]
    result = [round(x,2) for x in result]
    print('Attention_result:',result)

#    model_1= Model(inputs=model.input, outputs=model.get_layer(name="b_max_3").get_output_at(0))
    feat_1=model_1.predict(img)
    feat_1_average = np.average(feat_1, (0, 1, 2))

#    dim=len(np.shape(feat_1))
#    for i in range(0,dim-1):
#        feat_1_a = np.average(feat_1,0)
#    print(feat_1_a)

    feat_2 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    feat_2[0][atten_num] = 1
    print(feat_2)

    result = model_3.predict([feat_1, feat_2])
    result = result[0]
    result = softmax(result)
#    result = [round(x,2) for x in result]
    print('Attention modified result:', result)
#    attention_out = np.zeros(32)
    return result

for j in range(0,1):
    x_len = 64
    num_data = 1
    attention = np.zeros((7, num_data, x_len))
    init = 44
    for i in range(init, init + num_data):
        print("Anger")
        data_dir = "/data1/dataset/ferplus/train/Sadness/" + str(i)+ ".png"
        atten_out = attention_out(data_dir)
        attention[0, i - init, :len(atten_out)] = atten_out
        print("Happiness")
        data_dir = "/data1/dataset/ferplus/train/Happiness/" + str(i) + ".png"
        atten_out = attention_out(data_dir)
        attention[1, i - init, :len(atten_out)] = atten_out
        print(i)
        print("#######################")

    ang = np.average(attention[0, :, :], 0)
    hap = np.average(attention[1, :, :], 0)
    #neu = np.average(attention[2, :, :], 0)
    # for i in range(0,5):
    w = 0.4

    x_axis_1 = np.arange(0, x_len, 1)
    x_axis_2 = x_axis_1+w
    x_axis_3 = x_axis_1-w
    #plt = plt.subplot(111)
    label_1=plt.bar(x_axis_1, ang, width=w, label='Anger', color='g', alpha=0.3)
    label_2=plt.bar(x_axis_2, hap, width=w, label='Happiness', color='b', alpha=0.0)
    #plt.bar(x_axis_3, neu, width=w,label='Neutral', color='g', alpha=0.3)
    plt.ylim([0,5])

#    plt.legend([label_1], ["Sandess"])
#    plt.legend([label_1, label_2], ["Anger", "Happiness"])
#    plt.legend([label_1, label_2], ["Surprise", "Neutral"])
    plt.title('Attention result')
#    plt.xlabel('Attention value')
#    plt.title('results')
    plt.xlabel('# of channel')
    plt.ylabel('value')
    keys = [u'Anger', u'Happiness', u'Neutral', u'Sadness', u'Surprise']
#    plt.xticks(x_axis_1, keys, rotation='horizontal')

    fig = plt.gcf()
    plt.show()
    fig.savefig(logdir + 'attention_anger' + weight[:15] + '.png')

K.clear_session()
# print(result)

