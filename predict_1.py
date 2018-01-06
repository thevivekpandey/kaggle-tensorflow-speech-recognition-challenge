import numpy as np
from constants import PATH, LABELS, LABEL_2_INDEX, FINAL_I2L
import keras
from keras.models import Model
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from wrong_label_checker import WrongLabelChecker
import scipy.io.wavfile as wavfile
import librosa
import sys
import os

def describe(fullpath, n_mfcc, n_mels):
    arr, b = librosa.load(fullpath, sr=None)
    arr = np.append(arr, [0] * (16000 - len(arr)))
    S = librosa.feature.melspectrogram(arr, sr=16000, n_mels=n_mels)
    spec = librosa.power_to_db(S, ref=np.max)
    return np.array(arr), spec

def get_file(train_or_test, n_mfcc, n_mels):
    wlc = WrongLabelChecker()
    if train_or_test == 'test':
        base_path = '../downloads/test/audio/'
        files = os.listdir(base_path)
        for file in files:
            #label is file, so first argument is file
            yield file, describe(base_path + file, n_mfcc, n_mels)
    else:
        base_path = '../downloads/train/audio/'
        for label in LABELS:
            files = os.listdir(base_path + label)
            for file in files:
                #if wlc.is_labelled_wrongly(label + '/' + file):
                #    continue
                yield label + '/' + file, describe(base_path + label + '/' + file, n_mfcc, n_mels)

def get_shaped_input(train_or_test, n_mfcc, n_mels):
    dim1 = (16000, 1)
    dim2 = (n_mels, 32, 1)
    labels, output1, output2 = [], [], []
    for label, (arr, spec) in get_file(train_or_test, n_mfcc, n_mels):
        if len(labels) == 32:
            yield labels, np.array(output1), np.array(output2)
            labels, output = [], []
            output1, output2 = [], []
           
        labels.append(label)
        output1.append(arr.reshape(dim1))
        output2.append(spec.reshape(dim2))
    yield labels, np.array(output1), np.array(output2)

def one_model_prediction(train_or_test, model_name, params, output_file_1):
    n_mfcc = params['n_mfcc']
    n_mels = params['n_mels']

    json_file = open(model_name + '.json')
    model = model_from_json(json_file.read())
    json_file.close()
    model.load_weights(model_name + '.h5')
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model_intermediate = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
    output_file_1.write('fname,label\n')
    for labels, output1, output2 in get_shaped_input(train_or_test, n_mfcc, n_mels):
        ps = model.predict_on_batch([output1, output2])

        for l, p in zip(labels, ps[2]):
            output = FINAL_I2L[np.argmax(p)]
            output_file.write(l + ',' + output + '\n')

if __name__ == '__main__':
    train_or_test = sys.argv[1]
    model_name = sys.argv[2]
    assert train_or_test in ['train', 'test']
    assert len(sys.argv) == 3

    if train_or_test == 'test':
        output_file = open(model_name + '.out', 'w')
    else:
        output_file = open(model_name + '-train.out', 'w')

    params = {'n_mfcc': False, 'n_mels': 40}
    #params = {'n_mfcc': False, 'n_mels': False}
    one_model_prediction(train_or_test, model_name, params, output_file)
    output_file.close()
