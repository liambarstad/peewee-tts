import os
import numpy as np
import librosa
from torch.utils.data import Dataset

class SpeakerAudioDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 sample_rate, 
                 n_fft,
                 hop_length,
                 win_length,
                 n_mels):
        
        self.root_dir = root_dir
        
    def __getitem__(self, idx):
        pass
'''
class SpeakerAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate, mel_params):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.mel_params = mel_params
        self.utterances = []
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file[-3:] == 'wav':
                    info = file.split('_')
                    if len(info) == 4:
                        self.utterances.append([
                            info[0], info[1], info[2]+'_'+info[3]
                        ])
 
        # audio
        # | speaker_id | chapter_id | utterance_id | frame_id | ... 80 | 
        
        # text
        # | speaker_id | chapter_id | utterance_id | char_id | char_embed |
        
    def __len__(self):
        return len(self.utterances)
        # give length of all samples
        
    def __getitem__(self, idx):
        utterance = self.utterances[idx]
        y, _ = librosa.load(f'{self.root_dir}/{utterance[0]}/{utterance[1]}/{"_".join(utterance)}')
        mel_spec = librosa.feature.melspectrogram(y, sr=self.sample_rate, **self.mel_params)
        return utterance[0], mel_spec.swapaxes(0, 1)
'''
