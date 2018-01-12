from keras.callbacks import Callback
import os

class MyCallback(Callback):
    filenames = set()
    filenames_included_in_training_set = set()
    BASE_PATH = '../input/tensorflow-speech-recognition-challenge/test/audio/'
    def __init__(self, data_generator, model):
        print 'Initializing Callback'
        self.data_generator = data_generator
        self.model = model
        self.read_all_filenames()
        print 'Callback Initialized'

    def read_all_filenames(self):
        files = os.listdir(self.BASE_PATH)
        for file in files:
            self.filenames.add(file)

    def on_epoch_end(self, epoch, logs={}):
        print 'I see that epoch is ', epoch
        self.data_generator.add_data([0, 1], [[0]*16000, [0]*16000])
