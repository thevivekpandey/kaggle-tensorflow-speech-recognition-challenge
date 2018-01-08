from keras.callbacks import Callback
import os

class MyCallback(Callback):
    filenames = set()
    filenames_included_in_training_set = set()
    BASE_PATH = '../input/tensorflow-speech-recognition-challenge/test/audio/'
    def __init__(self, data_generator, model):
        self.data_generator = data_generator
        self.model = model
        self.read_all_filenames()

    def read_all_filenames(self):
        files = os.listdir(self.BASE_PATH)
        for file in files:
            self.filenames.add(file)
        print list(self.filenames)[1:10]

    def on_train_begin(self, logs={}):
        print 'ON TRAIN BEGIN'
        return
 
    #def on_train_end(self, logs={}):
    #    return
 
    def on_epoch_begin(self, epoch, logs={}):
        print 'ON EPOCH BEGIN'
        return
 
    def on_epoch_end(self, epoch, logs={}):
        print 'ON EPOCH END'
        return
 
    #def on_batch_begin(self, batch, logs={}):
    #    print 'Batch has begun'
    #    return
 
    def on_batch_end(self, batch, logs={}):
        #self.losses.append(logs.get('loss'))
        return
