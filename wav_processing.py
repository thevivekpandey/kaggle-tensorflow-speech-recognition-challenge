import sys
import numpy as np
import scipy.io.wavfile as wavfile

if __name__ == '__main__':
    filename = 'downloads/train/audio/bed/00176480_nohash_0.wav'
    rate, b = wavfile.read(filename)
    print rate, len(b)

    print 'Values'
    print b
    data_1 = np.load('processed/one_line_1d.npz')['data']
    print data_1
    data_2 = np.load('processed/one_line_2d.npz')['data']
    print data_2

    print 'Types'
    print type(b), type(data_1), type(data_2[0])
    print b.dtype, data_1.dtype, data_2[0].dtype

    print 'Shapes'
    print b.shape
    print data_1.shape
    print data_2[0].shape

    print 'Sums'
    print sum(b)
    print np.sum(b)
    print np.sum(data_1)
    print np.sum(data_2[0])

    print 'Equality'
    print np.array_equal(b, data_1)
    print np.array_equal(b, data_2[0])
    print np.array_equal(data_1, data_2[0])

    print 'More values'
    for i in [0, 1, 2, 3, 4, 101, 102, 103, 10001, 10002, 10003]:
        print b[i], data_1[i], data_2[0][i]
    wavfile.write('out.wav', rate, b)
    wavfile.write('out1d.wav', rate, data_1)
    wavfile.write('out2d.wav', rate, data_2[0])
