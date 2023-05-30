import random
import math
import librosa
import webrtcvad
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

class ToOrdinal:
    def __init__(self, char_values):
        self.char_values = char_values

    def __call__(self, data):
        return np.array([
            self.char_values.index(d.lower()) + 1
            for d in data
            if d.lower() in self.char_values
        ], dtype=int)

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

class VAD:
    def __init__(self, 
                 sample_rate: int, 
                 sample_rate_internal: int,
                 frame_size_ms: int, 
                 aggressiveness_index: int
                 ):
        self.sample_rate = sample_rate
        self.sample_rate_internal = sample_rate_internal
        self.frame_size_ms = frame_size_ms
        self.aggressiveness_index = aggressiveness_index

        self.vad = webrtcvad.Vad(self.aggressiveness_index)

    def __call__(self, data):
        output = np.array([])
        resampled = librosa.resample(data, orig_sr=self.sample_rate, target_sr=self.sample_rate_internal)
        frame_samp_size = int(self.sample_rate_internal * (self.frame_size_ms / 1000.0) * 2)
        pcm_data = self._float_to_pcm(resampled)
        total_frames = math.floor(len(pcm_data) / frame_samp_size)
        fr_approx_len = len(data) / total_frames
        for i in range(total_frames):
            frame = pcm_data[i*frame_samp_size:(i+1)*frame_samp_size]
            try:
                if self.vad.is_speech(frame, self.sample_rate_internal):
                    resampled_frame = data[math.floor(i*fr_approx_len):math.floor((i+1)*fr_approx_len)]
                    output = np.append(output, resampled_frame)
            except Exception as e:
                # I hate this package so much
                if e.__class__.__module__ == 'webrtcvad':
                    continue
                else:
                    raise e
        return output

    def _float_to_pcm(self, data):
        ints = (data * 32768).astype(np.int16)
        little_endian = ints.astype('<u2')
        return little_endian.tostring()
