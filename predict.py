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
    stdev = np.std(arr)
    S = librosa.feature.melspectrogram(arr, sr=16000, n_mels=n_mels)
    spec = librosa.power_to_db(S, ref=np.max)
    return arr/stdev, spec

def get_files(train_or_test, n_mfcc, n_mels):
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

def one_model_prediction(train_or_test, model_name, params, output_file_1, softmax_output_file_1, output_file_2, softmax_output_file_2, output_file_3, softmax_output_file_3):
    n_mfcc = params['n_mfcc']
    n_mels = params['n_mels']

    json_file = open(model_name + '.json')
    model = model_from_json(json_file.read())
    json_file.close()
    model.load_weights(model_name + '.h5')
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model_intermediate = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
    output_file_1.write('fname,label\n')
    output_file_2.write('fname,label\n')
    output_file_3.write('fname,label\n')
    dim1 = (1, 16000, 1)
    dim2 = (1, n_mels, 32, 1)
    for idx, (label, (arr, spec)) in enumerate(get_files(train_or_test, n_mfcc, n_mels)):
        a = np.array([arr])
        p = model.predict([a.reshape(dim1), spec.reshape(dim2)])

        output_1 = FINAL_I2L[np.argmax(p[0])]
        output_2 = FINAL_I2L[np.argmax(p[1])]
        output_3 = FINAL_I2L[np.argmax(p[2])]
        output_file_1.write(label + ',' + output_1 + '\n')
        softmax_output_file_1.write(label + '\t' + '\t'.join([str(x) for x in p[0][0]]) + '\t' + output_1 + '\n')
        output_file_2.write(label + ',' + output_2 + '\n')
        softmax_output_file_2.write(label + '\t' + '\t'.join([str(x) for x in p[1][0]]) + '\t' + output_2 + '\n')
        output_file_3.write(label + ',' + output_3 + '\n')
        softmax_output_file_3.write(label + '\t' + '\t'.join([str(x) for x in p[1][0]]) + '\t' + output_3 + '\n')

if __name__ == '__main__':
    train_or_test = sys.argv[1]
    model_name = sys.argv[2]
    assert train_or_test in ['train', 'test']
    assert len(sys.argv) == 3

    if train_or_test == 'test':
        output_file_1 = open(model_name + '-1.out', 'w')
        softmax_output_file_1 = open(model_name + '-1-softmax.out', 'w')
        output_file_2 = open(model_name + '-2.out', 'w')
        softmax_output_file_2 = open(model_name + '-2-softmax.out', 'w')
        output_file_3 = open(model_name + '-3.out', 'w')
        softmax_output_file_3 = open(model_name + '-3-softmax.out', 'w')
    else:
        output_file = open(model_name + '-train.out', 'w')
        softmax_output_file = open(model_name + '-train-softmax.out', 'w')

    params = {'n_mfcc': False, 'n_mels': 40}
    #params = {'n_mfcc': False, 'n_mels': False}
    one_model_prediction(train_or_test, model_name, params, output_file_1, softmax_output_file_1, output_file_2, softmax_output_file_2, output_file_3, softmax_output_file_3)
    output_file_1.close()
    softmax_output_file_1.close()
    output_file_2.close()
    softmax_output_file_2.close()
    output_file_3.close()
    softmax_output_file_3.close()
