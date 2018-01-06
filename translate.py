def tr(p):
   d = {
       'unknown': 'unknown',
       'silence': 'silence',
       'down': 'down',
       'yes': 'go',
       'stop': 'left',
       'no': 'no',
       'on': 'off',
       'right': 'on',
       'off': 'right',
       'go': 'stop',
       'left': 'up',
       'up': 'yes',
       'label': 'label'
   }
   return d[p]

res = {}
f = open('model-r1.out')
for line in f:
    part0 = line.strip().split(',')[0]
    part1 = line.strip().split(',')[1]
    res[part0] = tr(part1)
    print part0 + ',' + res[part0]
f.close()
