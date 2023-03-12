import math
import numpy as np
from scipy.signal import stft
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
            data,
            sr=self.sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=self.n_mels,
            window=self.window_function
        ))
        '''
        return [
            np.transpose(melspectrogram(
                data,
                sr=self.sample_rate,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=self.n_mels,
                window=self.window_function
            )) for utterance in data
        ]
        '''
