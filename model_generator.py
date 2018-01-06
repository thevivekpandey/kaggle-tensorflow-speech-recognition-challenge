import sys
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, Add, GlobalMaxPooling1D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, rmsprop
from keras.utils import np_utils
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras import regularizers

class ModelGenerator():
    def get_1d_part(self, i):
        strides = 2
        x = Convolution1D(filters=16, kernel_size=21, strides=strides, padding='same')(i)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(padding='same')(x)

        x = Convolution1D(filters=32, kernel_size=19, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(padding='same')(x)

        x1 = Flatten()(x)
        x1 = Dense(256, kernel_initializer='glorot_normal', activation='relu')(x1)
        x1 = Dropout(0.5)(x1)
        out1 = Dense(12, kernel_initializer='glorot_normal', activation='softmax')(x1)
       
        x = Convolution1D(filters=64, kernel_size=17, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(padding='same')(x)

        x = Convolution1D(filters=128, kernel_size=15, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(padding='same')(x)

        x2 = Flatten()(x)
        x2 = Dense(256, kernel_initializer='glorot_normal', activation='relu')(x2)
        x2 = Dropout(0.5)(x2)
        out2 = Dense(12, kernel_initializer='glorot_normal', activation='softmax')(x2)

        x = Convolution1D(filters=256, kernel_size=13, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(padding='same')(x)

        x = Convolution1D(filters=512, kernel_size=11, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(padding='same')(x)

        x = GlobalMaxPooling1D()(x)
        return out1, out2, x

    def get_mel_part(self, i):
        x = Conv2D(128, kernel_size=(3, 3), activation='relu')(i)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
       
        x1 = Flatten()(x)
        x1 = Dense(256, kernel_initializer='glorot_normal', activation='relu')(x1)
        x1 = Dropout(0.5)(x1)
        out1 = Dense(12, kernel_initializer='glorot_normal', activation='softmax')(x1)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

        x2 = Flatten()(x)
        x2 = Dense(256, kernel_initializer='glorot_normal', activation='relu')(x2)
        x2 = Dropout(0.5)(x2)
        out2 = Dense(12, kernel_initializer='glorot_normal', activation='softmax')(x2)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

        x = Flatten()(x)
        return out1, out2, x

    def get_overall_model(self, n_mels):
        raw_wav = Input(shape=(16000, 1))
        mel_spec = Input(shape=(n_mels, 32, 1))
        outa1, outa2, x = self.get_1d_part(raw_wav)
        outb1, outb2, y = self.get_mel_part(mel_spec)
        z = keras.layers.concatenate([x, y])
        z = Dropout(0.5)(z)
        z = Dense(256, activation='relu')(z)
        z = Dropout(0.5)(z)
        z = Dense(12, activation='softmax')(z)
        model = Model(inputs=[raw_wav, mel_spec], outputs=[outa1, outa2, outb1, outb2, z])
        return model

if __name__ == '__main__':
    n_mels = 40
    raw_wav = Input(shape=(16000, 1))
    mel_spec = Input(shape=(n_mels, 32, 1))
    mg = ModelGenerator()
    x = mg.get_1d_part(raw_wav)
    y = mg.get_mel_part(mel_spec)
    z = keras.layers.concatenate([x, y])
    z = Dropout(0.5)(z)
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.5)(z)
    z = Dense(12, activation='softmax')(z)
    model = Model(inputs=[raw_wav, mel_spec], outputs=[z])
    print model.summary()
