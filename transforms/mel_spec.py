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
        return int((self.sample_rate / 1000) * ms)

    def __call__(self, data):
        spec = melspectrogram(data[1], 
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                **self.mel_params
                ) 
        return data[0], spec
