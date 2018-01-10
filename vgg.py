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

class VGG():
    def ConvBn2d(self, i, f):
        x = Convolution2D(f, kernel_size=(3, 3), strides=(1, 1), padding='same')(i)
        x = BatchNormalization()(x)
        return x

    def vgg(self, n_mels):
        mel_spec = Input(shape=(n_mels, 32, 1)) 
        x = self.ConvBn2d(mel_spec, 8)
        x = Activation('relu')(x)
        x = self.ConvBn2d(x, 8)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = Dropout(0.2)(x)
        x = self.ConvBn2d(x, 16)
        x = Activation('relu')(x)
        x = self.ConvBn2d(x, 16)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        
        x = Dropout(0.2)(x)
        x = self.ConvBn2d(x, 32)
        x = Activation('relu')(x)
        x = self.ConvBn2d(x, 32)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(512)(x)
        x = Activation('relu')(x)
        
        x = Dropout(0.2)(x)
        x = Dense(256)(x)
        x = Activation('relu')(x)
        
        x = Dense(12, activation='softmax')(x)
        return Model(inputs=mel_spec, outputs=x)
