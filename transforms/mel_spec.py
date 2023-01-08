from librosa.feature import melspectrogram

class MelSpec:
    def __init__(self, 
                 sample_rate: int, 
                 hop_length_ms: int,
                 win_length_ms: int,
                 n_mels: int
                ):

        self.sample_rate = sample_rate
        self.hop_length_ms = hop_length_ms
        self.win_length_ms = win_length_ms
        self.n_mels = n_mels

        #'n_fft': int(1024 * (sample_rate / 16000)),
        #'hop_length': int(256 * (sample_rate / 16000)),
        #'win_length': sample_rate * window_len_ms / 1000
        #'win_length': int(1024 * (sample_rate / 16000)),

        self.n_fft = 
        self.win_length = self.sample_rate * window_le
        self.mel_params = kwargs

    def __call__(self, data):
        spec = melspectrogram(data[1], 
                sr=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels
                ) 
        return data[0], spec
