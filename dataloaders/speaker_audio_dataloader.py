from torch.utils.data import DataLoader

SpeakerAudioDataLoader = DataLoader
'''
class SpeakerAudioDataLoader(DataLoader):
    def __init__(self, dataset, N_speakers, M_utterances, **kwargs):
        DataLoader.__init__(self, dataset, **kwargs)
        self.N_speakers = N_speakers
        self.M_utterances = M_utterances
        print(self.dataset)

'''
