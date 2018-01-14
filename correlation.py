import sys

def get_scores(filename):
    f = open(filename)
    
    scores = {}
    for line in f:
        if 'fname' in line:
            continue
        parts = line.strip().split('\t')
        clip = parts[0]
        probs = [float(x) for x in parts[1:-1]]
        scores[clip] = probs
    f.close()
    return scores

def get_squared_difference(a1, a2):
    return sum([(x-y)**2 for (x, y) in zip(a1, a2)])

if __name__ == '__main__':
    models = sys.argv[1:]
    scores = {}
    for model in models:
        scores[model] = get_scores('models/model-' + model + '-softmax.out')

    clips = scores[models[0]].keys()
    corr = {}
    for m1 in models:
        for m2 in models:
            corr.setdefault(m1, {}).setdefault(m2, 0)
            for c in clips:
                corr[m1][m2] += get_squared_difference(scores[m1][c], scores[m2][c])

    for m1 in models:
        for m2 in models:
            corr[m1][m2] /= 158538.0

    print corr
