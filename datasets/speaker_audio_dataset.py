import random
import os
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SpeakerAudioDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 sources: dict,
                 m_utterances: int,
                 transform,
                 ):
        
        self.root_dir = root_dir
        self.sources = sources
        self.m_utterances = m_utterances
        self.transform = transform

        self.paths = pd.DataFrame([], columns=['dataset', 'speaker', 'path', 'speaker_id'])

        if 'LibriTTS' in sources:
            self._load_libritts(sources['LibriTTS'])
        
    def __len__(self):
        return self.paths.speaker_id.max() + 1

    def __getitem__(self, idx):
        # m_ut, partials_in_longest, n_mels, frames
        paths = self.paths[self.paths.speaker_id == idx].path
        speaker_data = []
        if len(paths.values) < self.m_utterances:
            # drop data where the number of utterances < M
            speaker_data = [ np.zeros(1) for _ in range(self.m_utterances) ]
        else:
            # select m utterances at random
            m_paths = random.sample(list(paths.values), self.m_utterances)
            for path in m_paths:
                data, _ = librosa.load(path)
                speaker_data.append(data)
        values = self.transform(speaker_data)
        labels = np.array([ idx for _ in range(self.m_utterances) ])
        return labels, values

    def _load_libritts(self, info={}):
        lbttsdir = os.path.join(self.root_dir, 'LibriTTS', info['version'])
        data = []
        for root, dirs, files in os.walk(lbttsdir):
            for file in files: 
                if file[-3:] == 'wav':
                    data.append(['LibriTTS', file.split('_')[0], os.path.join(root, file)])
        self._add_to_paths(data)

    def _add_to_paths(self, data):
        existing_data = self.paths[['dataset', 'speaker', 'path']]
        data = pd.DataFrame(data, columns=existing_data.columns)
        new_data = pd.concat([existing_data, data]).drop_duplicates()
        # add unique speaker_id column
        unique = new_data[['dataset', 'speaker']].drop_duplicates().reset_index().reset_index()
        unique = unique.rename(columns={unique.columns[0]:'speaker_id'})[['dataset', 'speaker', 'speaker_id']]
        self.paths = pd.merge(new_data, unique, on=['dataset', 'speaker'], how='left')

        return len(self.paths)


