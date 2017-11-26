import numpy as np
from constants import PATH, LABELS, LABEL_2_INDEX
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wavfile
import sys

print PATH
print LABELS

def read_n_lines(fname, n):
    arr = []
    f = open(fname)
    for i in range(n):
        line = f.readline()
        arr.append([int(t) for t in line.split(' ')])
    f.close()
    return arr
    
def get_data():
    x_data = np.load('processed/overall_int16.npz')['data']
    x_data = x_data / 32768.0
    y_data = np.load('processed/labels.npz')['data']
    num_training_examples = y_data.shape[0]
    num_categories = 30
    one_hot_y = np.zeros((num_training_examples, num_categories))
    print one_hot_y
    print y_data
    one_hot_y[np.arange(num_training_examples), y_data] = 1
    return x_data, one_hot_y

if __name__ == '__main__':
    print 'Loading data'
    x_data, y_data = get_data()
    #print x_data[0].dtype
    #wavfile.write('out.wav', 16000, x_data[0])
    #a, b = wavfile.read('out.wav')
    #print a
    #print b

    print 'Splitting data'
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10)
    x_train, y_train = x_data, y_data
    for i in [10000, 20000, 30000, 40000, 50000]:
        wavfile.write('out' + str(i) + '.wav', 16000, x_train[i] * 32768.0)
        print 'i = ', i
        print y_train[i]
    #sys.exit(1)
    #print x_data.shape
    #print x_train.shape
    #print x_test.shape
    print 'Enter Keras'
    model = Sequential()
    model.add(Dense(16, input_shape=(16000,)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    
    model.add(Dense(16))
    model.add(Activation('relu'))

    model.add(Dense(16))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    
    model.add(Dense(len(LABELS)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit(x_train, y_train, batch_size=32, nb_epoch=80, validation_data=(x_test, y_test))
