import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from random import randint
import librosa
import datetime
import acoustics
from wrong_label_checker import WrongLabelChecker

CATEGORY_2_LABEL = {
'one': 10,  'up': 0,     'bed': 10, 'two': 10, 'eight': 10, 'seven': 10, 'five': 10, 'nine': 10,
'no': 1,    'house': 10, 'left': 2,  'tree': 10,'down': 3,  'cat': 10,   'stop': 4,  'zero': 10,
'bird': 10, 'off': 5,    'three': 10,'right': 6, 'on': 7,   'marvin': 10,'four': 10, 'go': 8, 
'happy': 10,'wow': 10,   'sheila': 10,'yes': 9,  'six': 10, 'dog': 10}
NUM_CATEGORIES = len(set(CATEGORY_2_LABEL.values()))
BASE_PATH = '../input/tensorflow-speech-recognition-challenge/train/audio/'
categories = os.listdir(BASE_PATH)

class DataGenerator(object):
    # silence_too: I want to train fully with 12 categories
    # silence_vs_non_silence: I want to train with 2 categories: silence and no silence
    # If both the above are false: I want to train with 11 non silence categories
    def __init__(self, silence_too, silence_vs_non_silence, n_mfcc, n_mels):
        self.silence_too = silence_too
        self.silence_vs_non_silence = silence_vs_non_silence
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

        self.PROPORTIONS = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10] 
        assert(not(silence_vs_non_silence and silence_too))
        assert(not(n_mfcc and n_mels))
        if silence_vs_non_silence:
            self.PROPORTIONS.append(110)
        elif silence_too:
            self.PROPORTIONS.append(10)
        self.SUM = sum(self.PROPORTIONS)
        self.CUMUL = []
        for idx, prop in enumerate(self.PROPORTIONS):
            for i in range(prop):
                self.CUMUL.append(idx)

        self.data = self.get_training_data()
        self.silence_audio = self.get_silence_file()
        self.NOISE_IDX_2_COLOR= {0: 'white', 1: 'pink', 2: 'blue', 3: 'brown', 4: 'violet'}
        
    def shift_arr(self, arr, num):
        result = np.empty_like(arr)
        if num > 0:
            result[:num] = 0
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = 0
            result[:num] = arr[-num:]
        else:
            result = arr
        return result

    def generate(self, N, t):
        count = 0
        while True:
            count += 1
            
            #x, y = self.get_training_data_batch(N, t)
            x_raw, x_mel, y = self.get_training_data_batch(N, t)
            #if self.n_mfcc:
            #    yield x.reshape(x.shape[0], self.n_mfcc, 32, 1), y
            #elif self.n_mels:
            #    yield x.reshape(x.shape[0], self.n_mels, 32, 1), y
            #else:
            #    yield x.reshape(x.shape[0], 16000, 1), y
            yield [x_raw.reshape(x_raw.shape[0], 16000, 1), 
                   x_mel.reshape(x_mel.shape[0], self.n_mels, 32, 1)], y

    # I will give a batch of N, with labels follwing the above proportions
    # t is type: train or test: I will never give overlapping data for train and test. I will
    # give train data from first 90% of train data for any category and test data from
    # final 10% of same category
    def get_training_data_batch(self, N, t):
        # Get N random numbers to choose categories
        rands = np.random.randint(self.SUM, size=N)

        floats = np.random.random_sample(size=N)

        # fs are used to combine noise with signal. I don't give weightage of more than 0.25 to noise
        fs = np.random.random_sample(size=N) / 10

        # how much to shift each sample
        shifts = np.random.randint(-10, high=10, size=N)
   
        # With probablity 5%, use file. With probability 95%, use synthetic noise
        use_file = np.random.random_sample(size=N) < 0.05

        # noise colors
        noise_colors = np.random.randint(0, high=5, size=N)
        # Within the category, which element to choose
        assert t == 'train' or t == 'test'
        #x, y = [], []
        x_raw, x_mel, y = [], [], []
        for i in range(N):
            noise_color = self.NOISE_IDX_2_COLOR[noise_colors[i]]
            one_x_raw, one_x_mel, one_y = self.get_one_training_example(t, rands[i], floats[i], fs[i], shifts[i], use_file[i], noise_color)
            #one_x, one_y = self.get_one_training_example(t, rands[i], floats[i], 0, 0, use_file[i], noise_color)
            x_raw.append(one_x_raw) 
            x_mel.append(one_x_mel)
            y.append(one_y)
        
        #n_x = np.array(x)
        n_x_raw = np.array(x_raw)
        n_x_mel = np.array(x_mel)
        n_y = np.array(y)
        
        # silence is not counted in NUM_CATEGORIES
        #if self.silence_vs_non_silence:
        #    one_hot = np.zeros((N, 2))
        #elif self.silence_too:
        #    one_hot = np.zeros((N, NUM_CATEGORIES + 1))
        #else:
        #    one_hot = np.zeros((N, NUM_CATEGORIES))
        one_hot = np.zeros((N, NUM_CATEGORIES + 1))
        one_hot[np.arange(N), n_y] = 1
        #return n_x, one_hot
        return n_x_raw, n_x_mel, one_hot

    def get_one_training_example(self, t, one_rand, one_float, one_f, one_shift, use_file, noise_color):
        label = self.CUMUL[one_rand]
        if label == 11:
            # This is silence
            assert(self.silence_vs_non_silence or self.silence_too)
            if use_file:
                r = np.random.randint(len(self.silence_audio) - 16000)
                x = self.silence_audio[r:r+16000]
            else:
                x = acoustics.generator.noise(16000, color=noise_color) / 3
        else:
            cat_size = self.data[label].shape[0]
            f = 0.90 #What fraction of all data is for training
            if t == 'train':
                frac = one_float * f
            else:
                frac = f + one_float * (1 - f)
    
            idx = int(frac * cat_size) 
            # From category = label, I'll choose idx'th sample
            # I'll combine it with some noise
    
            if t == 'train':
                #r = np.random.randint(len(self.silence_audio) - 16000)
                #noise = self.silence_audio[r:r+16000]
                if use_file:
                    r = np.random.randint(len(self.silence_audio) - 16000)
                    noise = self.silence_audio[r:r+16000]
                else:
                    noise = acoustics.generator.noise(16000, color=noise_color) / 3
                 
                shifted_data = self.shift_arr(self.data[label][idx], one_shift)
                x = one_f * noise + (1 - one_f) * shifted_data
            else:
                x = self.data[label][idx]
            
        if self.silence_vs_non_silence:
            if label == 11:
                y = 1
            else:
                y = 0
        else:
            y = label

        #if self.n_mfcc:
        #    return librosa.feature.mfcc(y=x, sr=16000, n_mfcc=self.n_mfcc), y
        #elif self.n_mels:
        #    S = librosa.feature.melspectrogram(x, sr=16000, n_mels=self.n_mels)
        #    spec = librosa.power_to_db(S, ref=np.max)
        #    return spec, y
        #else:
        #    return x, y
        S = librosa.feature.melspectrogram(x, sr=16000, n_mels=self.n_mels)
        spec = librosa.power_to_db(S, ref=np.max)
        return x, spec, y
        
    def get_training_data(self):
        overall_x = {}
        count = 0
        wlc = WrongLabelChecker()
        for category in categories:
            temp_x = []
            if category == '_background_noise_':
                continue
            label = CATEGORY_2_LABEL[category]
            ws = os.listdir(BASE_PATH + category + '/')
            for w in ws:
                if wlc.is_labelled_wrongly(category + '/' + w):
                    continue
                if count % 5000 == 0:
                    print(count)
                count += 1
                arr, b = librosa.load(BASE_PATH + category + '/' + w, sr=None)
                arr = np.append(arr, [0] * (16000 - len(arr)))
                assert(len(arr) == 16000)
                temp_x.append(arr)
            if label not in overall_x:
                overall_x[label] = np.array(temp_x)
            else:
                # There are multiple directories for label 10
                overall_x[label] = np.append(overall_x[label], np.array(temp_x), axis=0)
        print("Saw " + str(count) + " files")
        
        return overall_x
    
    def get_silence_file(self):
        #Make one big silence file
        ws = os.listdir(BASE_PATH + '_background_noise_/')
        total_arr, b = librosa.load(BASE_PATH + '_background_noise_/running_tap.wav', sr=None)
        for w in ws:
            # colored noises are generated at runtime
            if w in ['README.md', 'running_tap.wav', 'pink_noise.wav', 'white_noise.wav']:
                continue
            arr, b = librosa.load(BASE_PATH + '_background_noise_/' + w, sr=None)
    
            total_arr = np.append(total_arr, arr)
        return total_arr
