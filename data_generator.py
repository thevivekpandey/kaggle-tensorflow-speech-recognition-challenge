import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from random import randint
import librosa
import datetime
import acoustics
from wrong_label_checker import WrongLabelChecker
import random

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
        #self.NOISE_IDX_2_COLOR= {0: 'white', 1: 'pink', 2: 'blue', 3: 'brown', 4: 'violet'} #blue, brown, voiolet noises have not helped
        self.NOISE_IDX_2_COLOR= {0: 'white', 1: 'pink'}
        
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
            
            x_raw, x_mel, y = self.get_training_data_batch(N, t)
            #if self.n_mfcc:
            #    yield x.reshape(x.shape[0], self.n_mfcc, 32, 1), y
            #if self.n_mels:
            #    yield [x_raw.reshape(x_raw.shape[0], 16000, 1), 
            #           x_mel.reshape(x_mel.shape[0], self.n_mels, 32, 1)], [y, y, y, y, y]
            if self.n_mels:
                yield x_mel.reshape(x_mel.shape[0], self.n_mels, 32, 1), y
            else:
                yield x_raw.reshape(x_raw.shape[0], 16000, 1), y

    # I will give a batch of N, with labels follwing the above proportions
    # t is type: train or test: I will never give overlapping data for train and test. I will
    # give train data from first 90% of train data for any category and test data from
    # final 10% of same category
    def get_training_data_batch(self, N, t):
        # Get N random numbers to choose categories
        rands = np.random.randint(self.SUM, size=N)

        floats = np.random.random_sample(size=N)

        #Whether or not to mix with noise
        mix_with_noise = np.random.random_sample(size=N) < 0.05

        # fs are used to combine noise with signal. I don't give weightage of more than 0.25 to noise
        fs = np.random.random_sample(size=N) / 10

        # how much to shift each sample
        shifts = np.random.randint(-10, high=10, size=N)
   
        # With probablity 15%, use file. With probability 85%, use synthetic noise
        use_file = np.random.random_sample(size=N) < 0.50

        # noise colors
        noise_colors = np.random.randint(0, high=2, size=N)

        # flip or not
        flip_or_not = np.random.randint(0, high=2, size=N)

        # zero silence: 1% of the time we give zero silence file
        zero_silence = np.random.random_sample(size=N) < 0.01

        # Within the category, which element to choose
        assert t == 'train' or t == 'test'
        #x, y = [], []
        x_raw, x_mel, y = [], [], []
        for i in range(N):
            noise_color = self.NOISE_IDX_2_COLOR[noise_colors[i]]
            one_x_raw, one_x_mel, one_y = self.get_one_training_example(t, rands[i], floats[i], mix_with_noise[i], fs[i], shifts[i], use_file[i], noise_color, flip_or_not[i], zero_silence[i])
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

    def get_one_training_example(self, t, one_rand, one_float, mix_with_noise, one_f, one_shift, use_file, noise_color, flip_or_not, zero_silence):
        label = self.CUMUL[one_rand]
        if label == 11:
            # This is silence
            assert(self.silence_vs_non_silence or self.silence_too)
            if zero_silence:
                x = np.zeros(16000)
            else:
                if use_file:
                    r = np.random.randint(len(self.silence_audio) - 16000)
                    x = self.silence_audio[r:r+16000]
                else:
                    x = acoustics.generator.noise(16000, color=noise_color) / 10
        else:
            cat_size = self.data[label].shape[0]
            # First 0.1 is for test, last 0.9 for training: by pseudo labelling, data
            # is appended at the end
            f = 0.10 
            if t == 'test':
                frac = one_float * f
            else:
                frac = f + one_float * (1 - f)
    
            # From category = label, I'll choose idx'th sample
            idx = int(frac * cat_size) 

            if use_file:
                r = np.random.randint(len(self.silence_audio) - 16000)
                noise = self.silence_audio[r:r+16000]
            else:
                noise = acoustics.generator.noise(16000, color=noise_color) / 30

            shifted_data = self.shift_arr(self.data[label][idx], one_shift)
            if mix_with_noise:
                x = one_f * noise + (1 - one_f) * shifted_data
            else:
                x = shifted_data
            
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
        assert(flip_or_not in (0, 1))
        if flip_or_not == 1:
            x = -x
        stdev = np.std(x)
        if stdev != 0:
            x = x/stdev
        S = librosa.feature.melspectrogram(x, sr=16000, n_mels=self.n_mels if self.n_mels else 40)
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
            random.shuffle(ws)
            
            for w in ws:
                if wlc.is_labelled_wrongly(category + '/' + w):
                    continue
                if count % 5000 == 0:
                    print(count)
                count += 1
                #if count % 100 !=0:
                #    continue
                arr, b = librosa.load(BASE_PATH + category + '/' + w, sr=16000)
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
        random.shuffle(ws)
        total_arr, b = librosa.load(BASE_PATH + '_background_noise_/running_tap.wav', sr=16000)
        for w in ws:
            # colored noises are generated at runtime
            if w in ['README.md', 'running_tap.wav', 'pink_noise.wav', 'white_noise.wav']:
                continue
            arr, b = librosa.load(BASE_PATH + '_background_noise_/' + w, sr=None)
    
            total_arr = np.append(total_arr, arr)
        return total_arr

    def print_num_data_points(self):
        for label in self.data:
            print str(label) + ': ' + str(len(self.data[label])),
        print 'silence:', str(len(self.silence_audio)),
        print

    def add_data(self, labels, np_arrs):
        print 'before-',
        self.print_num_data_points()
        for (label, np_arr) in zip(labels, np_arrs):
            #print 'adding to', label
            if label == 11:
                self.silence_audio = np.append(self.silence_audio, np_arr)
            else:
                self.data[label] = np.append(self.data[label], [np.array(np_arr)], axis=0)
        print 'after-',
        self.print_num_data_points()
