import numpy as np
from constants import PATH, LABELS, LABEL_2_INDEX, INDEX_2_NEW_INDEX
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D
from keras.layers import Conv1D
from keras.layers import Input, Dense
from keras.layers import Concatenate
import keras.layers
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wavfile
import sys

print PATH
print LABELS

def get_data():
    print 'Loading'
    x_data = np.load('../processed/overall_int16.npz')['data']
    print 'Dividing'
    x_data = x_data / 32768.0
    print 'Loading labels'
    y_data = np.load('../processed/labels.npz')['data']
    for i in range(len(y_data)):
        y_data[i] = INDEX_2_NEW_INDEX[y_data[i]]
    num_training_examples = y_data.shape[0]
    num_categories = 11
    one_hot_y = np.zeros((num_training_examples, num_categories))

    print one_hot_y
    print y_data
    one_hot_y[np.arange(num_training_examples), y_data] = 1
    return x_data, one_hot_y

def get_data_1():
    y_data = np.load('../processed/labels.npz')['data']
    for i in range(len(y_data)):
        y_data[i] = INDEX_2_NEW_INDEX[y_data[i]]
    num_training_examples = y_data.shape[0]
    num_categories = 11
    one_hot_y = np.zeros((num_training_examples, num_categories))
    one_hot_y[np.arange(num_training_examples), y_data] = 1
    print one_hot_y.shape

if __name__ == '__main__':
    #x_data, y_data = get_data()

    print 'Splitting data'
    #x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10)
    #x_train, y_train = x_data, y_data

    #for i in [10000, 20000, 30000, 40000, 50000]:
    #    wavfile.write('out' + str(i) + '.wav', 16000, x_train[i] * 32768.0)
    #    print 'i = ', i
    #    print y_train[i]
    
    print 'Enter Keras'
    #model = Sequential()
    #model.add(Dense(11, input_shape=(16000,)))
    #model.add(Activation('relu'))
    #
    #model.add(Dense(128))
    #model.add(Activation('relu'))

    #model.add(Dense(128))
    #model.add(Activation('relu'))
    #
    #model.add(Dense(128))
    #model.add(Activation('relu'))
    #
    #model.add(Dense(11))
    #model.add(Activation('softmax'))
    a = Input(shape=(16000,1))
    b = Convolution1D(filters=128, kernel_size=3, activation='relu')(a)
    c = Flatten()(b)

    a1 = Input(shape=(32, 16, 1))
    b1 = Convolution2D(129, kernel_size=(3,3), activation='relu')(a1)
    c1 = Flatten()(b1)
 
    d = keras.layers.concatenate([c, c1])
    d = Dense(128, activation='relu')(d)
    model = Model(inputs=[a, a1], outputs=d)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    print model.summary()
    #model.fit(x_train, y_train, batch_size=32, nb_epoch=6, validation_data=(x_test, y_test))
    #model.fit(x_data, y_data, batch_size=32, nb_epoch=1)

    #model_json = model.to_json()
    #with open('model-local-2.json', 'w') as f:
    #    f.write(model_json)
    #model.save_weights('model-local-2.h5')
    
    #model_yaml = model.to_yaml()
    #with open('model.yaml', 'w') as f:
    #    f.write(model_yaml)
    #model.save_weights('model.h5')
