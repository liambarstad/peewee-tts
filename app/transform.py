import math
import librosa
import numpy as np
from librosa.feature import melspectrogram

class ToOrdinal:
    def __init__(self, char_values):
        self.char_values = char_values

    def __call__(self, data):
        return np.array([
            self.char_values.index(d.lower()) + 1
            for d in data
            if d.lower() in self.char_values
        ], dtype=int)
    
class MelSpec:
    def __init__(self, 
                 sample_rate: int, 
                 hop_length_ms: int,
                 win_length_ms: int,
                 n_mels: int,
                 window_function='hann',
                 n_fft=2048
                ):

        self.sample_rate = sample_rate
        self.win_length_ms = win_length_ms
        self.hop_length_ms = hop_length_ms
        self.n_fft = n_fft
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
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            window=self.window_function
        ))
    
class InverseMelSpec:
    def __init__(self, sample_rate, hop_length_ms, win_length_ms, window_function='hann'):
        self.sample_rate = sample_rate
        self.hop_length_ms = hop_length_ms
        self.win_length_ms = win_length_ms
        self.window_function = window_function

    def __call__(self, mels):
        win_length = math.floor((self.win_length_ms / 1000) * self.sample_rate)
        hop_length = math.floor((self.hop_length_ms / 1000) * self.sample_rate) 
        return librosa.feature.inverse.mel_to_audio(
            M=mels,
            sr=self.sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            window=self.window_function
        )