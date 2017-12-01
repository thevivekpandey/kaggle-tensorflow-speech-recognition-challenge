import sys
import matplotlib
import matplotlib.pyplot as plt
from constants import LABELS
matplotlib.use('Agg')

if __name__ == '__main__':
    for label in LABELS:
        print label
        f = open('../processed/' + label + '.txt')
        for count, line in enumerate(f):
            print count
            arr_str = line.strip().split(' ')
            int_arr = [int(x) for x in arr_str]
            plt.plot(int_arr)
            plt.savefig('../processed/plots/' + label + '/'  + str(count) + '.png')
            plt.clf()
            if count == 10:
                sys.exit(1)
        print
        f.close()
