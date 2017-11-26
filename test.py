from constants import LABELS
import numpy as np
import time

name = 'processed/overall.npz'
print 'Loading ' + name
final = np.load(name)['data']

y = np.load('processed/labels.npz')['data']
print y
print y.shape
print final.shape
