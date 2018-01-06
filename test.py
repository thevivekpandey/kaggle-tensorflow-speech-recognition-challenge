import sys
LABELS = ['left', 'right', 'go', 'stop', 'on', 'off', 'up', 'down', 'yes', 'no']
LABELS_STR = 'bed bird cat dog down eight five four go happy house left marvin '
LABELS_STR += 'nine no off on one right seven sheila six stop three tree '
LABELS_STR += 'two up wow yes zero'

total, correct = 0, 0
f = open(sys.argv[1])
d = {}
#confusion[a][b] counts the number of cases where a is classified as b
confusion = {}
for line in f:
    actual, predicted = line.strip().split(',')
    actual = actual.split('/')[0]

    if actual == 'fname':
        continue
    d.setdefault(actual, {}).setdefault('total', 0)
    d.setdefault(actual, {}).setdefault('correct', 0)
    total += 1
    d[actual]['total'] += 1
    if actual in LABELS and actual == predicted:
        correct += 1
        d[actual]['correct'] += 1
    elif predicted == 'unknown':
        correct += 1
        d[actual]['correct'] += 1
    else:
        pass
        #print line.strip().split(',')[0]
    confusion.setdefault(actual, {}).setdefault(predicted, 0)
    confusion[actual][predicted] += 1
f.close()


for label, stats in d.iteritems():
    print label, '\t', 100.0 * stats['correct'] / stats['total']

#print confusion
ALL_LABELS = LABELS_STR.split(' ')
print 'CATEGORY\t' + '\t'.join(LABELS) + '\tunknown'
for actual in ALL_LABELS:
    print actual + '\t',
    for predicted in LABELS:
        print str(confusion[actual].get(predicted, 0)) + '\t',
    print confusion[actual].get('unknown', 0)

print 'total = ', total, 'correct = ', correct, 'ratio = ', 100.0 * correct / total
