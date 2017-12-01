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
    files = os.listdir('../downloads/test/audio')
    for idx, file in enumerate(files):
        if idx % 10000 == 0:
            print idx
        in_path = '../downloads/test/audio/' + file
        out_path = '../processed/test/' + file + '.txt'
        arr, size = describe(in_path)
        parr = arr.tolist()
        parr_str = [str(x) for x in arr] 
        f = open(out_path, 'w')
        f.write(' '.join(parr_str))
        f.close()
        #fullpath = PATH + label + '/' + file
        #arr, size = describe(fullpath)
        #if size == 16000:
        #    parr = arr.tolist()
        #    parr_str = [str(x) for x in arr]
        #    fx.write(' '.join(parr_str) + '\n')
        #    fy.write(' ' + '\n')
