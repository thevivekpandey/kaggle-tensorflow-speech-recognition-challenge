import os
import sys
import librosa
import matplotlib
import matplotlib.pyplot as plt
from constants import LABELS
matplotlib.use('Agg')

if __name__ == '__main__':
    for label in ['right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']:
        print label
        LABEL_PATH = '../downloads/train/audio/' + label + '/'
        fs = os.listdir(LABEL_PATH)
        for idx, f in enumerate(fs):
            if idx % 100 == 0:
                print idx
            arr, r = librosa.load(LABEL_PATH + f)

            plt.plot(arr)
            plt.savefig('../processed/plots/' + label + '/'  + f + '.png')
            plt.clf()
