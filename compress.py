import sys
import os
import gzip
import numpy as np
import scipy.io.wavfile as wavfile
from constants import PATH, LABELS

if __name__ == '__main__':
    arr = []
    for label in LABELS:
        print 'Working on ' + label
        f = open('processed/' + label + '.txt')
        for line in f:	
            str_arr = line.strip().split(' ')
            int_arr = [int(x) for x in str_arr]
            arr.append(int_arr)
        f.close()
        
        print 'Now len is ' + str(len(arr))
    print 'getting numpy array'
    narr = np.array(arr, dtype=np.int16)
    print 'got numpy array, saving it'
    np.savez_compressed('processed/overall_int16.npz', data=narr)
    print 'done saving'
   
