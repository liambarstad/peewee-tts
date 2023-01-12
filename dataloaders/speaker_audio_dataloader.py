from torch.utils.data import DataLoader

class SpeakerAudioDataLoader(DataLoader):
    def __init__(self, 
            dataset, 
            **kwargs
            ):
        DataLoader.__init__(self, dataset, **kwargs)
        self.N_speakers = N_speakers
        self.M_utterances = M_utterances
        self.partial_utterance_length_ms = partial_utterance_length_ms

    def collate(self, data):
        # for sample in data:
        #   y = sample[0]
        #   x = sample[1]
        return SpeakerBatch(data)




