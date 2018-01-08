import sys
from constants import FINAL_I2L

def predict_1(score1, score2):
    # Check if model2 is predicting silence. If so, just output silence
    max2_label = -1 
    max2_prob = 0
    for label, prob in score2.iteritems():
        if prob > max2_prob:
            max2_prob = prob
            max2_label = label
    if max2_label == 11:
        return FINAL_I2L[11]

    # Check if model2 is > 0.8 sure. If yes, go with its choice
    if max2_prob > 0.8:
        return FINAL_I2L[max2_label]

    max1_label = -1 
    max1_prob = 0
    for label, prob in score1.iteritems():
        if prob > max1_prob:
            max1_prob = prob
            max1_label = label
    if max1_prob > 0.9:
        return FINAL_I2L[max1_label]

    # Else let's take sum of probabilties. For score2, we will distribute
    # The prob for silence, but we will do that later
    arr = []
    for i in range(11):
        #we will normalize score2
        normalized = score2[i] + score2[11] * (score2[i] / (1 - score2[11]))
        normalized *= 1.03
        arr.append(score1[i] + normalized)

    max_label = -1
    max_prob = 0
    for i in range(11):
        if arr[i] > max_prob:
            max_label = i
            max_prob = arr[i]
    return FINAL_I2L[max_label]
       
def predict_2(score1, score2):
    # Check if model2 is predicting silence. If so, just output silence
    max2_label = -1 
    max2_prob = 0
    for label, prob in score2.iteritems():
        if prob > max2_prob:
            max2_prob = prob
            max2_label = label
    if max2_label == 11:
        return FINAL_I2L[11]

    max1_label = -1 
    max1_prob = 0
    for label, prob in score1.iteritems():
        if prob > max1_prob:
            max1_prob = prob
            max1_label = label

    if max1_prob > 0.8 and max1_label == 10 and max2_prob < max1_prob + 0.1:
        return 'unknown'
    if max2_prob > 0.8 and max2_label == 10 and max1_prob < max2_prob + 0.1:
        return 'unknown'
    if max1_prob < 0.5 and max2_prob < 0.5 and (max1_label == 10 or max2_label == 10):
        return 'unknown'
    # Else let's take sum of probabilties. For score2, we will distribute
    # The prob for silence, but we will do that later
    arr = []
    for i in range(11):
        #we will normalize score2
        normalized = score2[i] + score2[11] * (score2[i] / (1 - score2[11]))
        normalized *= 1.03
        arr.append(score1[i] + normalized)

    max_label = -1
    max_prob = 0
    for i in range(11):
        if arr[i] > max_prob:
            max_label = i
            max_prob = arr[i]
    return FINAL_I2L[max_label]
       
def predict_3(score1, score2):
    # Check if model2 is predicting silence. If so, just output silence
    max2_label = -1 
    max2_prob = 0
    for label, prob in score2.iteritems():
        if prob > max2_prob:
            max2_prob = prob
            max2_label = label
    if max2_label == 11:
        return FINAL_I2L[11]

    # Else let's take sum of probabilties. For score2, we will distribute
    # The prob for silence, but we will do that later
    arr = []
    for i in range(11):
        #we will normalize score2
        normalized = score2[i] + score2[11] * (score2[i] / (1 - score2[11]))
        normalized *= 1.02
        arr.append(score1[i]**0.8 + normalized**0.8)

    max_label = -1
    max_prob = 0
    for i in range(11):
        if arr[i] > max_prob:
            max_label = i
            max_prob = arr[i]
    return FINAL_I2L[max_label]
       
print 'fname,label'
dict1, dict2 = {}, {}
f1_name = sys.argv[1] #1d conv: does not contain silence prediction
f2_name = sys.argv[2] #mel model: contains silence prediction

f1 = open(f1_name)
for line in f1:
    if line.strip() == 'fname,label':
        continue
    parts = line.strip().split('\t')
    clip = parts[0]
    nums = [float(s) for s in parts[1:-1]]
    dict1[clip] = {}
    for idx, num in enumerate(nums):
        dict1[clip][idx] = num
f1.close()

f2 = open(f2_name)
for line in f2:
    if line.strip() == 'fname,label':
        continue
    parts = line.strip().split('\t')
    clip = parts[0]
    nums = [float(s) for s in parts[1:-1]]
    dict2[clip] = {}
    for idx, num in enumerate(nums):
        dict2[clip][idx] = num
f2.close()

for clip, scores in dict2.iteritems():
    print clip + ',' + predict_3(dict1[clip], dict2[clip])
