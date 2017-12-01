import numpy as np
from constants import PATH, LABELS, LABEL_2_INDEX, FINAL_I2L
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wavfile
import sys
import os

def describe(fullpath):
    a, b = wavfile.read(fullpath)
    return np.array(b), len(b)

def get_files(train_or_test):
    if train_or_test == 'test':
        base_path = '../downloads/test/audio/'
        files = os.listdir(base_path)
        for file in files:
            #label is file, so first argument is file
            yield file, describe(base_path + file)
    else:
        base_path = '../downloads/train/audio/'
        for label in LABELS:
            files = os.listdir(base_path + label)
            for file in files:
                yield label + '/' + file, describe(base_path + label + '/' + file)

def get_training_data(normalize):
    CATEGORY_2_LABEL = {
    'one': 10,
    'up': 0,
    'bed': 10,
    'two': 10,
    'eight': 10,
    'seven': 10,
    'five': 10,
    'nine': 10,
    'no': 1,
    'house': 10,
    'left': 2,
    'tree': 10,
    'down': 3,
    'cat': 10,
    'stop': 4,
    'zero': 10,
    'bird': 10,
    'off': 5, 
    'three': 10,
    'right': 6,
    'on': 7, 
    'marvin': 10,
    'four': 10, 
    'go': 8, 
    'happy': 10,
    'wow': 10, 
    'sheila': 10,
    'yes': 9,
    'six': 10,
    'dog': 10
    }
    NUM_CATEGORIES = len(set(CATEGORY_2_LABEL.values()))
    print('Num categories = ', NUM_CATEGORIES)
    BASE_PATH = '../downloads/train/audio/'
    categories = os.listdir(BASE_PATH)
    overall_x, overall_y = [], []
    for category in categories:
        if category == '_background_noise_':
            continue
        ws = os.listdir(BASE_PATH + category + '/')
        for w in ws:
            arr, l = describe(BASE_PATH + category + '/' + w)
            if l != 16000:
                continue
            overall_x.append(arr)
            overall_y.append(CATEGORY_2_LABEL[category])
    
    print('done reading the files')
    if normalize:
        n_x = np.array(overall_x) / 32768.0
    else:
        n_x = np.array(overall_x)
    print('done normalizing')
    n_y = np.array(overall_y)
    
    one_hot = np.zeros((n_y.shape[0], NUM_CATEGORIES))
    one_hot[np.arange(n_y.shape[0]), n_y] = 1
    print(n_x.shape)
    print(n_y.shape)
    return n_x, one_hot
    print 'fname,label'
    for idx, (label, (arr, l)) in enumerate(get_files(train_or_test)):
        if l != 16000:
            print label + ',-1'
        else:
            p = model.predict(np.array([arr]))
            print label + ',' + FINAL_I2L[np.argmax(p)]

if __name__ == '__main__':
    model_name = sys.argv[1]
    json_file = open(model_name + '.json')
    model = model_from_json(json_file.read())
    json_file.close()
    model.load_weights(model_name + '.h5')

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    x_data, y_data = get_training_data(True)
    r = model.evaluate(x=x_data, y=y_data)
    
