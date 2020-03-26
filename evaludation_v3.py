from sklearn.metrics import classification_report
import numpy as np
from keras import models
import os
from load_image_v3 import fer_load_data
import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model = models.load_model("/data3/younghoon/Flagship_2019/Multimodal_2019/image_5emo/data/1217_1609/1217_1609.h5")

x_train, x_test, y_train, y_test = fer_load_data()
y_train=y_train.tolist()
for x in range(0,7):
    print(y_train.count(x))
y_train = keras.utils.to_categorical(y_train, num_classes=7)
y_test = keras.utils.to_categorical(y_test, num_classes=7)

y_pred = model.predict(x_train)
y_classes = y_pred.argmax(axis=-1)
print(y_classes)
y_classes=y_classes.tolist()
for x in range(0,7):
    print(y_classes.count(x))
