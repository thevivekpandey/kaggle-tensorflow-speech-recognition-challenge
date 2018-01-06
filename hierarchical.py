import sys
print 'fname,label'

models = sys.argv[1:]
silence_model = sys.argv[1]
full_model = sys.argv[2]
wav_2_model_2_labels = {}
for model in models:
    f = open(str(model) + '.out')
    for line in f:
        wav, label = line.strip().split(',')
        if wav == 'fname':
            continue
        wav_2_model_2_labels.setdefault(wav, {})[model] = label
    f.close()

for wav, model_2_labels in wav_2_model_2_labels.iteritems():
    if model_2_labels[silence_model] == 'silence':
        print wav + ',' + 'silence'
    else:
        print wav + ',' + model_2_labels[full_model]
