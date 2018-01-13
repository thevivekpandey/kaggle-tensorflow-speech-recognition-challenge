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
 
def predict_4(scores):
    arr = []
    for i in range(12):
        #arr.append(sum([score[i] ** 0.5 for score in scores]))
        arr.append((scores[0][i] * 0.98)**0.5 + scores[1][i]**0.5 + scores[2][i]**0.5)

    max_label = -1
    max_prob = 0
    for i in range(12):
        if arr[i] > max_prob:
            max_label = i
            max_prob = arr[i]
    return FINAL_I2L[max_label]
    
def predict_5(scores):
    arr = []
    for i in range(12):
        arr.append((scores[1][i])**2.0 + scores[2][i]**2.0 + scores[3][i]**2.0)

    max_label = -1
    max_prob = 0
    for i in range(12):
        if arr[i] > max_prob:
            max_label = i
            max_prob = arr[i]
    #If first 3 guys rule silence, silence it is
    if max_label == 11:
        return FINAL_I2L[max_label]
    
    #If first 3 guys rule something other than silence, we vote again
    arr = []
    for i in range(11):
        #arr.append(sum([score[i] ** 0.5 for score in scores]))
        part0 = scores[0][i] * 0.99
        part1 = scores[1][i] / (1 - scores[1][11])
        part2 = scores[2][i] / (1 - scores[2][11])
        part3 = scores[3][i] / (1 - scores[3][11])
        arr.append(part0**0.1 + part1**0.1 + part2**0.1 + part3**0.1)

    max_label = -1
    max_prob = 0
    for i in range(11):
        if arr[i] > max_prob:
            max_label = i
            max_prob = arr[i]
    return FINAL_I2L[max_label]
    



def load_scores_from_file(fname):
    my_dict = {}
    f = open(fname)
    for line in f:
        if line.strip() == 'fname,label':
            continue
        parts = line.strip().split('\t')
        clip = parts[0]
        nums = [float(s) for s in parts[1:-1]]
        my_dict[clip] = {}
        for idx, num in enumerate(nums):
            my_dict[clip][idx] = num
    f.close()
    return my_dict

if __name__ == '__main__':
    dict1, dict2 = {}, {}
    fnames = sys.argv[1:]
    scores = {}
    
    for f in fnames:
        scores[f] = load_scores_from_file(f)
 
    print 'fname,label'
    clips = scores[fnames[0]].keys()
    for clip in clips:
        one_clip_score = [scores[f][clip] for f in fnames] 
        print clip + ',' + predict_5(one_clip_score)
