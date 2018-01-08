import sys
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
from data_generator import DataGenerator
from model_generator import ModelGenerator
from my_callback import MyCallback
from heng_cher_keng_model_generator import HengCherKengModelGenerator

def get_conv_model_1():
    model = Sequential()
    
    strides = 2
    model.add(Convolution1D(filters=16, kernel_size=21, strides=strides, padding='same', input_shape=(16000,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(LeakyReLU())
    model.add(MaxPooling1D(padding='same'))

    model.add(Convolution1D(filters=32, kernel_size=19, strides=strides, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(LeakyReLU())
    model.add(MaxPooling1D(padding='same'))

    model.add(Convolution1D(filters=64, kernel_size=17, strides=strides, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(LeakyReLU())
    model.add(MaxPooling1D(padding='same'))

    model.add(Convolution1D(filters=128, kernel_size=15, strides=strides, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(LeakyReLU())
    model.add(MaxPooling1D(padding='same'))

    model.add(Convolution1D(filters=256, kernel_size=13, strides=strides, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(LeakyReLU())
    model.add(MaxPooling1D(padding='same'))

    model.add(Convolution1D(filters=512, kernel_size=11, strides=strides, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(LeakyReLU())
    model.add(MaxPooling1D(padding='same'))

    #model.add(Convolution1D(filters=1024, kernel_size=9, strides=strides, padding='same'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(MaxPooling1D(padding='same'))

    #model.add(Flatten())
    model.add(GlobalMaxPooling1D())
    
    model.add(Dense(256, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(12))
    model.add(Activation('softmax'))

    return model

def add_conv_layers(inp, depth):
    conv = Convolution1D(filters=2^(10-depth), 
                         kernel_size=17, 
                         strides=2, 
                         padding='same', 
                         input_shape=(16000, 1))(inp)
    conv = Activation('relu')(conv)
    conv = MaxPooling1D()(conv)

    for i in range(depth - 1):
        conv = Convolution1D(filters=2^(10-depth) * 2^i, kernel_size=15 - 2*i, strides=2, padding='same')(conv)
        conv = Activation('relu')(conv)
        conv = MaxPooling1D()(conv)
        
    #conv = Flatten()(conv)
    conv = GlobalMaxPooling1D()(conv)
    return conv

def get_conv_model_2():
    inp = Input(shape=(16000,1))
    convs = [add_conv_layers(inp, 1), 
             add_conv_layers(inp, 2),
             add_conv_layers(inp, 3),
             add_conv_layers(inp, 4),
             add_conv_layers(inp, 5),
             add_conv_layers(inp, 6)
            ]
    out = Concatenate()(convs)
    conv_model = Model(input=inp, output=out)

    model = Sequential()
    model.add(conv_model)
    model.add(Dense(512, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(11))
    model.add(Activation('softmax'))

    return model

def get_mel_model(silence_vs_non_silence, silence_too, n_mels):

    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(n_mels, 32, 1)))
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

    assert(not(silence_vs_non_silence and silence_too))
    if silence_vs_non_silence:
        model.add(Dense(2, activation='softmax'))
    elif silence_too:
        model.add(Dense(12, activation='softmax'))
    else:
        model.add(Dense(11, activation='softmax'))
    return model
    
    #model = Sequential()
    #model.add(Conv2D(128, kernel_size=(15,15), activation='relu', input_shape=(n_mels, 32, 1)))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(12, activation='softmax'))
    #return model
    
def get_mfcc_model(n_mfcc):
    model = Sequential()
    
    model.add(Conv2D(128, (3, 3), activation = 'relu', padding= 'same', input_shape=(n_mfcc, 32, 1)))
    model.add(Conv2D(128, (3, 3), activation = 'relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation = 'relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation = 'relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation = 'relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation = 'relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(192, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(192, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(12, activation = 'softmax'))
    return model

def run_keras(model, model_number, n_mfcc, n_mels, silence_vs_non_silence, silence_too):
    generator = DataGenerator(silence_vs_non_silence=silence_vs_non_silence, silence_too=silence_too, n_mfcc=n_mfcc, n_mels=n_mels)
    training_generator = generator.generate(128, 'train')
    test_generator = generator.generate(128, 'test')

    opt = Adam(lr=0.001, decay=0)
    #opt = keras.optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam', loss_weights=[1.0])
    #model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    filepath = "models/model-" + model_number + "-{epoch:03d}-{val_dense_2_acc:.4f}-{val_dense_4_acc:.4f}-{val_dense_6_acc:.4f}-{val_dense_8_acc:.4f}-{val_dense_10_acc:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, mode='max')
    reduce_lr = ReduceLROnPlateau(verbose=1, min_lr = 1e-8, patience=5, factor=0.3)
    log_callback = MyCallback(generator, model)
    callbacks = [checkpoint, reduce_lr, log_callback]


    model.fit_generator(generator=training_generator, validation_data=test_generator, 
                        steps_per_epoch=20, validation_steps=2,
                        epochs=200,
                        callbacks=callbacks)
    return model

model_number = sys.argv[1]
silence_vs_non_silence = False
silence_too = True
n_mfcc=False
#n_mels = 40
n_mels = False
#model = get_mel_model(silence_vs_non_silence=silence_vs_non_silence, silence_too=silence_too, n_mels=n_mels)
#model = get_conv_model_1()
#model = ModelGenerator().get_overall_model(n_mels)
model = HengCherKengModelGenerator().get_1d_conv_model(n_mels)

print model.summary()
model_json = model.to_json()
with open('models/model-' + model_number + '.json', 'w') as f:
    f.write(model_json)
model = run_keras(model, model_number, n_mfcc=n_mfcc, n_mels=n_mels, silence_vs_non_silence=silence_vs_non_silence, silence_too=silence_too)
model.save_weights('models/model-' + model_number + '.h5')
