PATH = 'downloads/train/audio/'
LABELS_STR = 'bed bird cat dog down eight five four go happy house left marvin '
LABELS_STR += 'nine no off on one right seven sheila six stop three tree '
LABELS_STR += 'two up wow yes zero'
LABELS = LABELS_STR.split(' ')
LABEL_2_INDEX = {}

FINAL_I2L = {
    0: 'up',
    1: 'no',
    2: 'left',
    3: 'down',
    4: 'stop',
    5: 'off',
    6: 'right',
    7: 'on',
    8: 'go',
    9: 'yes',
    10: 'unknown',
    11: 'silence'
}

#RAHUL_I2L = {
#    0: 
#    1:
#    2:
#    3:
#    4:
#    5:
#    6:
#    7:
#    8:
#    9:
#    10: 'unknown',
#    11: 'silence'
#}
for i, label in enumerate(LABELS):
    LABEL_2_INDEX[label] = i

# bed-0, bird-1, cat-2, dog-3, down-4, eight-5, five-6, four-7
# go-8, happy-9, house-10, left-11, marving-12, nine-13, no-14
# off-15, on-16, one-17, right-18, seven-19, sheila-20, six-21
# stop-22, three-23, tree-24, two-25, up-26, wow-27, yes-28
# zero-29
# We deal with 11 labels: 
# 0-yes, 1-no, 2-up, 3-down, 4-left, 5-right, 
# 6-on, 7-off, 8-stop, 9-go, 10-unknown
INDEX_2_NEW_INDEX = {
   0: 10,
   1: 10,
   2: 10,
   3: 10,
   4: 3,
   5: 10,
   6: 10,
   7: 10,
   8: 9,
   9: 10,
   10: 10,
   11: 4,
   12: 10,
   13: 10,
   14: 1,
   15: 7,
   16: 6,
   17: 10,
   18: 5,
   19: 10,
   20: 10,
   21: 10,
   22: 8,
   23: 10,
   24: 10,
   25: 10,
   26: 2,
   27: 10,
   28: 0,
   29: 10
}
