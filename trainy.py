from constants import LABEL_2_INDEX, LABELS
import numpy as np

arr_y = []
for label in LABELS:
    f = open('processed/' + label + '.txt')
    print 'Processing ' + label
    for line in f:
        arr_y.append(LABEL_2_INDEX[label])
    f.close()

narr = np.array(arr_y)
np.savez_compressed('processed/labels.npz', data=narr)
