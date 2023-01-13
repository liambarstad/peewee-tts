from torch.utils.data import DataLoader

class SpeakerBatch:
    def __init__(self, data):
        self
        import ipdb; ipdb.set_trace()
        pass


class SpeakerAudioDataLoader(DataLoader):
    def __init__(self, 
            dataset, 
            m_utterances,
            **kwargs
            ):
        DataLoader.__init__(self, dataset, **kwargs)
        self.m_utterances = m_utterances

    def collate(self, data):
        # get shortest in batch
        #   
        # for sample in data:
        #   y = sample[0]
        #   x = sample[1]
        #
        # divide into 800ms segments

        return SpeakerBatch(data)




