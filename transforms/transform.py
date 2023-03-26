import random
import math
import numpy as np
import noisereduce as nr
from librosa.feature import melspectrogram

class OneHotEncodeCharacters:
    def __init__(self, values: str):
        self.values = values

    def __call__(self, text):
        encoded = np.array([[]])
        for t in text:
            item = np.zeros(len(self.values))
            item[self.values.index(t)] = 1.
            if encoded.shape[1] == 0:
                encoded = np.append(encoded, item.reshape(1, -1), axis=1)
            else:
                encoded = np.append(encoded, item.reshape(1, -1), axis=0)
        return encoded.astype(int)

class ReduceNoise:
    def __init__(self,
                 sample_rate: int,
                 prop_decrease: float,
                 n_fft: int,
                 n_jobs=1
                 ):
        self.sample_rate = sample_rate
        self.prop_decrease = prop_decrease
        self.n_fft = n_fft
        self.n_jobs = n_jobs

    def __call__(self, data):
        reduced = nr.reduce_noise(
            y=data,
            sr=self.sample_rate,
            prop_decrease=self.prop_decrease,
            n_fft=self.n_fft,
            n_jobs=self.n_jobs
        ) 
        reduced[np.isnan(reduced)] = 0.0
        return reduced.astype(float)

class MelSpec:
    def __init__(self, 
                 sample_rate: int, 
                 hop_length_ms: int,
                 win_length_ms: int,
                 n_mels: int,
                 window_function='hann'
                ):

        self.sample_rate = sample_rate
        self.win_length_ms = win_length_ms
        self.hop_length_ms = hop_length_ms
        self.n_mels = n_mels
        self.window_function = window_function

    def __call__(self, data):
        win_length = math.floor((self.win_length_ms / 1000) * self.sample_rate)
        hop_length = math.floor((self.hop_length_ms / 1000) * self.sample_rate)
        return np.transpose(melspectrogram(
            y=data,
            sr=self.sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=self.n_mels,
            window=self.window_function
        ))

class ParsePartial:
    def __init__(self, t, indexof=0):
        self.t = t
        self.indexof = indexof

    def __call__(self, data):
        # get partial utterance with each frame length t
        if data.shape[0] < self.t:
            padding = np.zeros([self.t - data.shape[0], data.shape[1]])
            return np.append(data, padding, axis=0)
        else:
            random_ind = random.randint(0, data.shape[0] - self.t)
            return data[random_ind:random_ind+self.t, :]
