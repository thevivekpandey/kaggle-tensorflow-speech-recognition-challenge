from keras.callbacks import Callback
import os
import random
from datetime import datetime

NUM_SAMPLES_TO_PROBE = 5000
MIN_PROB_TO_ADD = 0.99
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
        if epoch < 5:
            return

        t1 = datetime.now()
        samples_to_probe = random.sample(self.filenames_remaining, NUM_SAMPLES_TO_PROBE)
        labels, probs, arrs = self.prediction_engine.predict(self.model, samples_to_probe)
        t2 = datetime.now()
        print
        print 'Time to predict:', (t2-t1)

        f = open('models/lists-' + self.model_name + '-' + str(epoch) + '.out', 'w')
        filtered_labels, filtered_arrs = [], []
        count = 0
        for i in range(len(labels)):
            if probs[i] > MIN_PROB_TO_ADD:
                count += 1
                f.write(samples_to_probe[i] + '\t' + str(labels[i]) + '\t' + str(probs[i]) + '\n')
                filtered_labels.append(labels[i])
                filtered_arrs.append(arrs[i])
                self.filenames_remaining.remove(samples_to_probe[i])
      
        print 'Found ', count, 'eligible samples of which silences were', len([x for x in filtered_labels if x == 11])
        print 'After epoch', epoch, len(self.filenames_remaining), 'labels remain'
        f.close()
        self.data_generator.add_data(filtered_labels, filtered_arrs)
