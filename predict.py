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
    return np.array(b) / 32768.0, len(b)

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

if __name__ == '__main__':
    model_name = sys.argv[1]
    train_or_test = sys.argv[2]
    assert train_or_test in ['train', 'test']

    json_file = open(model_name + '.json')
    model = model_from_json(json_file.read())
    json_file.close()
    model.load_weights(model_name + '.h5')

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    print 'fname,label'
    for idx, (label, (arr, l)) in enumerate(get_files(train_or_test)):
        if l != 16000:
            print label + ',-1'
        else:
            p = model.predict(np.array([arr]))
            print label + ',' + FINAL_I2L[np.argmax(p)]
