import sys
import os
import numpy as np
import scipy.io.wavfile as wavfile
from constants import LABELS
from constants import LABEL_2_INDEX

def describe(fullpath):
    a, b = wavfile.read(fullpath)
    return b, len(b)

'''
This script processes all the wav files, and writes their constituent array
in a file. One file is created for each lable, so there are ~30 files
created. Files are named <label>.txt.
'''
if __name__ == '__main__':
    x = []
    y = []
    for label in LABELS:
        files = os.listdir(PATH + label)
        fx = open('processed/examples.txt', 'w')
        fy = open('processed/lables.txt', 'w')
        for file in files:
            fullpath = PATH + label + '/' + file
            arr, size = describe(fullpath)
            if size == 16000:
                parr = arr.tolist()
                parr_str = [str(x) for x in arr]
                fx.write(' '.join(parr_str) + '\n')
                fy.write(' ' + '\n')
        fx.close()
        fy.close()
        print 'Done with ' + label
