import os
import sys
import librosa
import matplotlib
import matplotlib.pyplot as plt
from constants import LABELS
matplotlib.use('Agg')

def plot_train():
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

def plot_test():
    BASE_PATH = '../downloads/test/audio/'
    OUTPUT_PATH = '../processed/test-plots/'
    fs = os.listdir(BASE_PATH)
    for idx, f in enumerate(fs):
        if idx % 100 == 0:
            print idx
        arr, r = librosa.load(BASE_PATH + f, sr=None)
        plt.figure(figsize=(5, 5))
        plt.plot(arr, '-')
        plt.savefig(OUTPUT_PATH + f + '.png')
        plt.clf()

if __name__ == '__main__':
    plot_test()
