class WrongLabelChecker:
    wrong_examples = set()
    def __init__(self):
        f = open('wrong_labels.txt')
        for line in f:
            self.wrong_examples.add(line.strip())
        f.close()

    def is_labelled_wrongly(self, example):
        return example in self.wrong_examples

if __name__ == '__main__':
    wlc = WrongLabelChecker()
    print wlc.is_labelled_wrongly('down/190821dc_nohash_3.wav')
    print wlc.is_labelled_wrongly('hi')
