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
        return encoded

class MelSpec:
    def __init__(self, 
                 sample_rate: int, 
                 hop_length_ms: int,
                 win_length_ms: int,
                 n_mels: int,
                 **kwargs
                ):

        self.sample_rate = sample_rate
        self.n_fft = self._to_frames(win_length_ms)
        self.hop_length = self._to_frames(hop_length_ms)
        self.n_mels = n_mels
        self.mel_params = kwargs

    def _to_frames(self, ms):
        samples_per_millisecond = 0.001 * self.sample_rate
        return int(samples_per_millisecond * ms)

    def __call__(self, data):
        specs = [
            np.transpose(
                melspectrogram(
                utterance, 
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                **self.mel_params
            )) for utterance in data
        ]
        return specs

class STFT:
    def __init__(self,
                 sample_rate: int,
                 frame_size_ms: int,
                 frame_hop_ms: float,
                 window_function: str
                ):
        self.sample_rate = sample_rate
        self.frame_size_ms = frame_size_ms
        self.frame_hop_ms = frame_hop_ms
        self.window_function = window_function

    def __call__(self, data):
        nperseg = math.floor(self.sample_rate * (self.frame_size_ms / 1000))
        noverlap = (self.frame_hop_ms / 1000) * self.sample_rate
        data, _, _ = stft(data, 
                    nperseg=nperseg,
                    noverlap=noverlap,
                    window=self.window_function
                )
        return data

