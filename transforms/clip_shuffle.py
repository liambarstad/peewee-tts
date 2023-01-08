class ClipShuffle:
    def __init__(self,
                 sample_rate: int,
                 fixed_length_ms: int
                ):

        self.sample_rate = sample_rate
        self.fixed_length_ms = fixed_length_ms

    def __call__(self, data):
        return data
