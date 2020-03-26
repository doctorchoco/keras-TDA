import glob
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.model_selection import train_test_split


def fer_load_data():
    data_dir = "/data1/dataset/ferplus/train/"
    emo_list = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    train_data = np.zeros((count,48,48,3))
    train_label = np.zeros(count)

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            train_data[count] = img.astype('float32') / 255
            train_label[count] = emo
            count = count + 1

    data_dir = "/data1/dataset/ferplus/test/"
    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    test_data = np.zeros((count,48,48,3))
    test_label = np.zeros(count)

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            test_data[count] = img.astype('float32') / 255
            test_label[count] = emo
            count = count + 1


    return train_data, test_data, train_label, test_label

def fer_5emo_load_data():
    data_dir = "/data1/dataset/ferplus/train/"
    emo_list = ['Anger', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    train_data = np.zeros((count,48,48,3))
    train_label = np.zeros(count)

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            train_data[count] = img.astype('float32') / 255
            train_label[count] = emo
            count = count + 1

    data_dir = "/data1/dataset/ferplus/test/"
    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    test_data = np.zeros((count,48,48,3))
    test_label = np.zeros(count)

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            test_data[count] = img.astype('float32') / 255
            test_label[count] = emo
            count = count + 1

    return train_data, test_data, train_label, test_label

def fer_ang_load_data():
    data_dir = "/data1/dataset/ferplus/train/"
    emo_list = ['Anger']

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    train_data = np.zeros((count,48,48,3))
    train_label = np.zeros(count)

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            train_data[count] = img.astype('float32') / 255
            train_label[count] = emo
            count = count + 1

    data_dir = "/data1/dataset/ferplus/test/"
    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    test_data = np.zeros((count,48,48,3))
    test_label = np.zeros(count)

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            test_data[count] = img.astype('float32') / 255
            test_label[count] = emo
            count = count + 1

    return train_data, test_data, train_label, test_label

def fer_hap_load_data():
    data_dir = "/data1/dataset/ferplus/train/"
    emo_list = ['Happiness']

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    train_data = np.zeros((count,48,48,3))
    train_label = np.zeros(count)

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            train_data[count] = img.astype('float32') / 255
            train_label[count] = emo
            count = count + 1

    data_dir = "/data1/dataset/ferplus/test/"
    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    test_data = np.zeros((count,48,48,3))
    test_label = np.zeros(count)

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            test_data[count] = img.astype('float32') / 255
            test_label[count] = emo
            count = count + 1

    return train_data, test_data, train_label, test_label

def fer_neu_load_data():
    data_dir = "/data1/dataset/ferplus/train/"
    emo_list = ['Neutral']

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    train_data = np.zeros((count,48,48,3))
    train_label = np.zeros((count,5))

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            train_data[count] = img.astype('float32') / 255
            train_label[count] = [0, 0, 1, 0, 0]
            count = count + 1

    data_dir = "/data1/dataset/ferplus/test/"
    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    test_data = np.zeros((count,48,48,3))
    test_label = np.zeros((count,5))

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            test_data[count] = img.astype('float32') / 255
            test_label[count] = [0, 0, 1, 0, 0]
            count = count + 1

    return train_data, test_data, train_label, test_label

def fer_sad_load_data():
    data_dir = "/data1/dataset/ferplus/train/"
    emo_list = ['Sad']

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    train_data = np.zeros((count,48,48,3))
    train_label = np.zeros((count,5))

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            train_data[count] = img.astype('float32') / 255
            train_label[count] = [0, 0, 0, 1, 0]
            count = count + 1

    data_dir = "/data1/dataset/ferplus/test/"
    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    test_data = np.zeros((count,48,48,3))
    test_label = np.zeros((count,5))

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            test_data[count] = img.astype('float32') / 255
            test_label[count] = [0, 0, 0, 1, 0]
            count = count + 1

    return train_data, test_data, train_label, test_label

def fer_sur_load_data():
    data_dir = "/data1/dataset/ferplus/train/"
    emo_list = ['Surprise']

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    train_data = np.zeros((count,48,48,3))
    train_label = np.zeros((count,5))

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            train_data[count] = img.astype('float32') / 255
            train_label[count] = [0, 0, 0, 0, 1]
            count = count + 1

    data_dir = "/data1/dataset/ferplus/test/"
    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        count = count + len(img_paths)

    test_data = np.zeros((count,48,48,3))
    test_label = np.zeros((count,5))

    count = 0
    for emo in range(0, len(emo_list)):
        img_paths = glob.glob(data_dir + emo_list[emo]+ "/*.png", recursive=True)
        for j in range (0,len(img_paths)):
            img=img_to_array(load_img(img_paths[j]))
            test_data[count] = img.astype('float32') / 255
            test_label[count] = [0, 0, 0, 0, 1]
            count = count + 1

    return train_data, test_data, train_label, test_label

