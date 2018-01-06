from __future__ import print_function
import numpy as np # linear algebra
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from scipy import signal
import os
import random
from scipy.io import wavfile
import librosa
import sys

batch_size = 256
num_classes = 12
epochs = 30
sample_rate = 16000
labels = 'yes no up down left right on off stop go unknown silence'
unknowns = 'bed bird cat dog eight five four happy house marvin nine one seven sheila six three tree two wow zero'

def spectrogram(samples):
    eps=1e-10
    frequencies, times, spectrogram = signal.stft(samples, sample_rate, nperseg = sample_rate/50, noverlap = sample_rate/75)
    return np.log(np.abs(spectrogram).T+eps)

files = []
for dir_name, _, file_names in os.walk('../downloads/train/audio/'):
    file_names = [file_name for file_name in file_names if file_name.endswith('.wav')]
    for file_name in file_names:
        files.append(os.path.join(dir_name, file_name))
lf = []
uf = []
for file_name in files:
    if any([unknown for unknown in unknowns.split() if unknown in file_name]):
        uf.append(file_name)
    else:
        lf.append(file_name)
random.shuffle(uf)
print('len of labeled files: ', len(lf))
print('len of unknown files:', len(uf))
print('sample lf file: ', lf[:1])
print('smaple uf file: ', uf[:1])
uf = uf[:13000]
wav_files = lf + uf
print('len of wav files: ' + str(len(wav_files)))
random.shuffle(wav_files)
wav_files = wav_files[:4000]
#for wav_file in wav_files[:10]:
    #graph_spectrogram(wav_file, wav_file)

#exit(0)
print(wav_files[0].split('/'))

labels_to_predict = labels.split()
X_train_orig, Y_train_orig = [], []
X_test_orig, Y_test_orig = [], []
files_with_sample_rate_less_than_16000 = []
num_unknowns = 0
num_labels = 0
num_silence = 0
label_count_dict = {}
unknowns = unknowns.split()
vp_count = 0
vp_inside_count = 0
vp_outside_count = 0
print("Numbr f wav files = " + str(len(wav_files)))
for wav_file in wav_files:
    label = wav_file.split('/')[-2]
    _, samples = wavfile.read(wav_file)
    vp_outside_count += 1
    for i in range(0, len(samples), sample_rate):
        vp_inside_count += 1
        if len(samples[i:]) < sample_rate:
            diff = sample_rate - len(samples[i:])
            diff_div = diff // 2
            samples_to_consider = samples[i:]
            samples_to_consider = np.lib.pad(samples_to_consider, (diff_div, diff - diff_div), 'constant', constant_values = (0, 0))
        else:
            samples_to_consider = samples[i : i + sample_rate]
        #spec = spectrogram(samples_to_consider)
        S = librosa.feature.melspectrogram(samples_to_consider, sr=sample_rate, n_mels=40)
        spec = librosa.power_to_db(S, ref=np.max)
        if len(X_test_orig) < 3000:
            X_test_orig.append(spec)
            if label in unknowns:
                num_unknowns += 1
                Y_test_orig.append(labels_to_predict.index('unknown'))
                label_count_dict.setdefault('unknown', 0)
                label_count_dict['unknown'] += 1
            elif label == '_background_noise_':
                num_silence += 1
                label_count_dict.setdefault('silence', 0)
                label_count_dict['silence'] += 1
                Y_test_orig.append(labels_to_predict.index('silence'))
            else:
                num_labels += 1
                if labels_to_predict.index(label) == None:
                    print('label not in labels_to_predict: ', label)
                Y_test_orig.append(labels_to_predict.index(label))
                label_count_dict.setdefault(label, 0)
                label_count_dict[label] += 1
            continue

        vp_count += 1
        X_train_orig.append(spec)
        if label in unknowns:
            num_unknowns += 1
            Y_train_orig.append(labels_to_predict.index('unknown'))
            label_count_dict.setdefault('unknown', 0)
            label_count_dict['unknown'] += 1
        elif label == '_background_noise_':
            num_silence += 1
            label_count_dict.setdefault('silence', 0)
            label_count_dict['silence'] += 1
            Y_train_orig.append(labels_to_predict.index('silence'))
        else:
            num_labels += 1
            if labels_to_predict.index(label) == None:
                print('label not in labels_to_predict: ', label)
            Y_train_orig.append(labels_to_predict.index(label))
            label_count_dict.setdefault(label, 0)
            label_count_dict[label] += 1
        if len(X_train_orig) % 10000 == 0:
            print('len of X_train_orig: ', len(X_train_orig))

print("vp_count = " + str(vp_count))
print("vp_inside_count = " + str(vp_inside_count))
print("vp_outside_count = " + str(vp_outside_count))
print(X_train_orig.shape)
sys.exit(1)
X_train = np.array(X_train_orig).astype(np.float32)
X_test = np.array(X_test_orig).astype(np.float32)
Y_train = keras.utils.to_categorical(np.array(Y_train_orig).astype(np.float32), num_classes)
Y_test = keras.utils.to_categorical(np.array(Y_test_orig).astype(np.float32), num_classes)
print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
print('num_unknowns: ', num_unknowns)
print('num_labels: ', num_labels)
print('num_silence: ', num_silence)
print('num files_with_sample_rate_less_than_16000: ', len(files_with_sample_rate_less_than_16000))


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
print('X_train shape: ', X_train.shape)
input_shape = (X_train.shape[1], X_train.shape[2], 1)
print('printing label count dict')
for key, value in label_count_dict.items():
    print(key, value)


# model = Sequential()
# model.add(Conv1D(16, 128, strides=16, activation='relu', input_shape=(sample_rate, 1)))
# model.add(Conv1D(16, 64, strides=16, activation='relu'))
# #model.add(Conv1D(64, 64, activation='relu'))
# model.add(MaxPooling1D(pool_size=2, strides=2))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.RMSprop(),
#               metrics=['accuracy'])

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
checkpoint = ModelCheckpoint('full_model.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
callback_list = [checkpoint]
model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callback_list,
          validation_data=(X_test, Y_test),
          verbose=1)
