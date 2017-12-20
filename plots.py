import os
import sys
import librosa
import matplotlib
import matplotlib.pyplot as plt
from constants import LABELS
matplotlib.use('Agg')

if __name__ == '__main__':
    for label in ['right']:
        print label
        LABEL_PATH = '../downloads/train/audio/' + label + '/'
        fs = os.listdir(LABEL_PATH)
        for idx, f in enumerate(fs):
            if idx % 100 == 0:
                print idx
            arr, r = librosa.load(LABEL_PATH + f)

            
            plt.figure(figsize=(200,7))
            plt.plot(arr, '-')
            plt.savefig('../processed/plots1/' + label + '/'  + f + '.png')
            plt.clf()
