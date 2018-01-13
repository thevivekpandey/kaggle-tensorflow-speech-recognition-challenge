from constants import PATH, LABELS, LABEL_2_INDEX, FINAL_I2L
import librosa
import numpy as np

class PredictionEngine:
    def __init__(self, n_mfcc, n_mels):
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

    def describe(self, fullpath):
        arr, b = librosa.load(fullpath, sr=16000)
        arr = np.append(arr, [0] * (16000 - len(arr)))
        stdev = np.std(arr)
        if stdev != 0:
            arr = arr / stdev
        S = librosa.feature.melspectrogram(arr, sr=16000, n_mels=self.n_mels)
        spec = librosa.power_to_db(S, ref=np.max)
        return np.array(arr), spec

    def predict(self, model, filenames):
        base_path = '../input/tensorflow-speech-recognition-challenge/test/audio/'
        labels, probs = [], []
        arrs = []
        for filename in filenames:
            arr, spec = self.describe(base_path + filename)
            dim2 = (1, self.n_mels, 32, 1)
            p = model.predict(spec.reshape(dim2))
            labels.append(np.argmax(p))
            probs.append(np.amax(p))
            arrs.append(np.array(arr))
            print filename, np.argmax(p), np.amax(p)
        return labels, probs, arrs
