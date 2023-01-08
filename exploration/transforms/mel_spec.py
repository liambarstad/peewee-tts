from librosa.feature import melspectrogram

class MelSpec:
    def __init__(self, 
                 sample_rate: int, 
                 **kwargs
                ):

        self.sample_rate = sample_rate
        self.mel_params = kwargs

    def __call__(self, data):
        return data[0], melspectrogram(data[1], sr=self.sample_rate, **self.mel_params) 
