gpu_num = input('GPU Number:')

#### Library ####
import os
import keras.layers
from keras.callbacks import EarlyStopping
from keras.optimizers import adam
import time

from keras import callbacks
from keras import backend as K
import numpy as np
from keras import preprocessing
import cv2
from load_image_v3 import*
from image_attention_v3 import*
from sklearn.utils import class_weight
import numpy as np
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.optimizers import SGD

from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from imblearn.keras import BalancedBatchGenerator
import sys
import tensorflow as tf
import shutil

####################################################################################################################

def save_options(file_path, current_time, optimizer, learning_rate, score, loss, loops, atten_layer, base_model_load):
    file = open(file_path +current_time+'.txt', 'w')
    file.write("Model_name: %s\n" % current_time)
    file.write("Optimizer: %s\n" % optimizer)
    file.write("Learning rate: %s\n" % learning_rate)
    file.write("loss: %s\n" % loss)
    file.write("Score: %s\n" % score)
    file.write("Loops: %s\n" % loops)
    file.write("Attention_layer: %s\n" % atten_layer)
    file.write("Base_model_load: %s\n" % base_model_load)
    file.close()


def init(gpu_num):
    ### GPU Setting
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num


def main(log_folder,current_time):
    ### Set options
    optimizer = "s"
    learning_rate = 0.01
    loops = 1
    atten_layer = 6
    base_model_load = 1
#    loss = 'mean_squared_error'
    loss = 'categorical_crossentropy'
    init(gpu_num)
    x_train, x_test, y_train, y_test = fer_5emo_load_data()

    print('Data Load Completed!')
    ### Convert labels to categorical one-hot encoding
    num_class = 5
#    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
#    class_weights = [0.1, 0.1, 0.1, 0.1, 1]
#    print(class_weights)
#    class_weights_valid = class_weight.compute_class_weight('balanced', np.unique(y_test), y_test)
#    print(class_weights_valid)

    y_train = keras.utils.to_categorical(y_train, num_classes=num_class)
    y_test = keras.utils.to_categorical(y_test, num_classes=num_class)

    datagen = ImageDataGenerator(
#        featurewise_center=True,
#        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True)

#    training_generator, steps_per_epoch = BalancedBatchGenerator(x_train, y_train, batch_size=256, random_state=42)

    ######################################################################################################

    ### Parameter setting
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 30},gpu_options=gpu_options)
    sess = tf.Session(config=config)
    K.set_session(sess)

    ## Model Load ##
    gpu_model = integration_model_v1(loops, atten_layer)
    #    weights_list = VR_model.get_weights()

    with open(log_folder + current_time + '_model.txt', 'w') as fh:
        gpu_model.summary(print_fn=lambda x: fh.write(x + '\n'))
        try:
            gpu_model.get_layer(name="attention_model").summary(print_fn=lambda x: fh.write(x + '\n'))
        except:
            print("No attention")

    ### Train
    if optimizer == 'a':
        opt = keras.optimizers.Adam(lr=float(learning_rate), beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001, amsgrad=False)
    elif optimizer == 's':
        opt = keras.optimizers.SGD(lr=float(learning_rate), momentum=0.0, decay=0.0, nesterov=False)
    elif optimizer == 'r':
        opt = keras.optimizers.RMSprop(lr=float(learning_rate), rho=0.9)

    ## Current base model
    if base_model_load == 1:
#        gpu_model.load_weights("/data3/younghoon/Flagship_2019/Multimodal_2019/image_5emo/data/1217_1202/weights.100-1.45.h5",by_name=True)
        gpu_model.load_weights("/data3/younghoon/Flagship_2019/Multimodal_2019/image_5emo/data/1217_1202/1217_1202.h5",by_name=True)
#        gpu_model.load_weights("/data3/younghoon/Flagship_2019/Multimodal_2019/image_5emo/data/1217_1312/weights.200-1.24.h5",by_name=True)

    # Base model fix
    if (loops > 0):
        for layer in gpu_model.layers[0:13]:
            layer.trainable = False
        for l in gpu_model.layers:
            print(l.name, l.trainable)
        print(loops)
    else :
        print(loops)

#    gpu_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    gpu_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    ## Callbacks
#    early_stopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=50, verbose=1, mode='auto')

    tb_hist = keras.callbacks.TensorBoard(log_dir=log_folder, histogram_freq=1, write_graph=False, write_images=True)

#    lr_show = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
#    history=gpu_model.fit_generator(generator=training_generator, steps_per_epoch= 128, validation_data= (x_test, y_test) , verbose=2, epochs=64, callbacks=[tb_hist])
#    history=gpu_model.fit_generator(datagen.flow(x_train, y_train, batch_size=16), steps_per_epoch=len(x_train) / 16, validation_data= (x_test, y_test) , verbose=2, epochs=64, callbacks=[tb_hist])
    model_check=keras.callbacks.ModelCheckpoint(log_folder+'weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=False,
                                              save_weights_only=False, mode='auto', period=10)

    ## Training
    history = gpu_model.fit(x_train, y_train, batch_size = 10, initial_epoch = 1,epochs = 300, validation_data= [x_test, y_test], verbose=2, callbacks=[tb_hist, model_check], shuffle='batch')
#    history = gpu_model.fit(x_train, y_train, batch_size = 100, initial_epoch = 2,epochs = 300, validation_data= [x_test, y_test], verbose=2, callbacks=[tb_hist, model_check], shuffle='batch')
    ## Model Save
    model_name = log_folder+ current_time + '.h5'
    print(model_name)
    gpu_model.save(model_name)
    score = gpu_model.evaluate(x_test, y_test, batch_size = 256)

    ## Write parameters
    save_options(log_folder, current_time, optimizer, learning_rate, score, loss, loops, atten_layer, base_model_load)

    ## Plotting training and validation curve
    history_dict = history.history
    history_dict.keys()

    import matplotlib.pyplot as plt

    acc = history.history['acc']
#    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    # ‘bo’blue dot
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # ‘b’blue line
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss'+current_time)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    fig= plt.gcf()
    fig.savefig(log_folder + current_time + '.png', dpi=300)
    plt.legend()
    plt.show()

    K.clear_session()

if __name__ == '__main__':
    current_time = str(time.localtime().tm_mon).zfill(2) + str(time.localtime().tm_mday).zfill(2) + '_' + str(time.localtime().tm_hour).zfill(2) + str(time.localtime().tm_min).zfill(2)
    print(current_time)

    log_folder = '/data3/younghoon/Flagship_2019/Multimodal_2019/image_5emo/data/'+ current_time + "/"

    if not (os.path.isdir(log_folder)):
        os.makedirs(os.path.join(log_folder))
    try:
        main(log_folder, current_time)
    except KeyboardInterrupt:
        if (os.path.isdir(log_folder)):
            shutil.rmtree(os.path.join(log_folder))
    print("End")