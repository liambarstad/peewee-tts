import numpy as np

class ClipShuffle:
    def __init__(self,
                 sample_rate: int,
                 win_length_ms: int,
                 fixed_length_ms: int
                ):

        self.sample_rate = sample_rate
        self.win_length_ms = win_length_ms
        self.fixed_length_ms = fixed_length_ms

    # TODO: move all into single file
    def _to_frames(self, ms):
        samples_per_millisecond = 0.001 * self.sample_rate
        return int(samples_per_millisecond * ms)

    def __call__(self, data):
        # return all full subdivisions of data by fixed length in ms
        sample_length = self.sample_rate * (self.fixed_length_ms / 1000)
        num_partials = int((len(data[1]) * 1000) / (self.fixed_length_ms * self.sample_rate))
        partial_utterances = []
        for x in range(num_partials):
            ms_per_second = self.fixed_length_ms / 1000
            x0 = ms_per_second * x * self.sample_rate
            x1 = ms_per_second * (x+1) * self.sample_rate
            partial_utterances.append(data[1][int(x0):int(x1)])

        return data[0], np.array(partial_utterances) 
