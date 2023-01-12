import numpy as np
from librosa.feature import melspectrogram

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
            melspectrogram(
                partial_utterance, 
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                **self.mel_params
            ) for partial_utterance in data[1]
        ]
        return data[0], np.array(specs)
