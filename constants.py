PATH = 'downloads/train/audio/'
LABELS_STR = 'bed bird cat dog down eight five four go happy house left marvin '
LABELS_STR += 'nine no off on one right seven sheila six stop three tree '
LABELS_STR += 'two up wow yes zero'
LABELS = LABELS_STR.split(' ')
LABEL_2_INDEX = {}

for i, label in enumerate(LABELS):
    LABEL_2_INDEX[label] = i
