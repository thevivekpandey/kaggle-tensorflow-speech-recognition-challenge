from keras.callbacks import Callback
import os
import random

NUM_SAMPLES_TO_PROBE = 10
MIN_PROB_TO_ADD = 0
class MyCallback(Callback):
    filenames_gone_to_training = set()
    filenames_remaining = set()
    BASE_PATH = '../input/tensorflow-speech-recognition-challenge/test/audio/'
    def __init__(self, data_generator, prediction_engine, model_name):
        print 'Initializing Callback'
        self.data_generator = data_generator
        self.prediction_engine = prediction_engine
        self.model_name = model_name
        self.read_all_filenames()
        print 'Callback Initialized'

    def read_all_filenames(self):
        files = os.listdir(self.BASE_PATH)
        for file in files:
            self.filenames_remaining.add(file)

    def on_epoch_end(self, epoch, logs={}):
        print 'I see that epoch is ', epoch

        samples_to_probe = random.sample(self.filenames_remaining, NUM_SAMPLES_TO_PROBE)
        labels, probs, arrs = self.prediction_engine.predict(self.model, samples_to_probe)

        f = open('models/lists-' + self.model_name + '-' + str(epoch) + '.out', 'w')
        filtered_labels, filtered_arrs = [], []
        for i in range(len(labels)):
            if probs[i] > MIN_PROB_TO_ADD:
                f.write(samples_to_probe[i] + '\t' + str(labels[i]) + '\n')
                filtered_labels.append(labels[i])
                filtered_arrs.append(arrs[i])
                self.filenames_remaining.remove(samples_to_probe[i])
      
        print 'After epoch', epcoh, len(self.filenames_remaining), 'labels remain'
        f.close()
        self.data_generator.add_data(filtered_labels, filtered_arrs)
