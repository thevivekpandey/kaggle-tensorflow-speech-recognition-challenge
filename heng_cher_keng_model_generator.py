import sys
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, Add, GlobalMaxPooling1D, Conv2D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, rmsprop
from keras.utils import np_utils
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras import regularizers

class HengCherKengModelGenerator():
    def __init__(self):
        pass

    def get_1d_conv_model(self, n_mels):
        raw_wav = Input(shape=(16000, 1))
        x = Convolution1D(filters=8, kernel_size=3, padding='same')(raw_wav)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution1D(filters=8, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

        x = Convolution1D(filters=16, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution1D(filters=16, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

        x = Convolution1D(filters=32, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution1D(filters=32, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
        x = Dropout(0.1)(x)

        x = Convolution1D(filters=64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution1D(filters=64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
        x = Dropout(0.1)(x)

        x = Convolution1D(filters=128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution1D(filters=128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
        x = Dropout(0.2)(x)

        x = Convolution1D(filters=256, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution1D(filters=256, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
        x = Dropout(0.2)(x)

        x = Convolution1D(filters=256, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution1D(filters=256, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
        x = Dropout(0.2)(x)

        x = Convolution1D(filters=512, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution1D(filters=512, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
        x = Dropout(0.2)(x)

        x = Convolution1D(filters=512, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution1D(filters=512, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
        x = Dropout(0.2)(x)

        x = AveragePooling1D()(x)
        x = Flatten()(x)

        x = Dropout(0.5)(x)
        x = Dense(512, kernel_initializer='glorot_normal', activation='relu')(x)
        x = Activation('relu')(x)

        x = Dropout(0.5)(x)
        x = Dense(256, kernel_initializer='glorot_normal', activation='relu')(x)
        x = Activation('relu')(x)

        x = Dense(12, activation='softmax')(x)
        return Model(inputs=raw_wav, outputs=x)
