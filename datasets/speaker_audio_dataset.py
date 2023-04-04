import random
import librosa
import soundfile
import numpy as np
import pandas as pd
from .dataset import Dataset

class SpeakerAudioDataset(Dataset):
    def __init__(self, 
                 source: str,
                 repos: dict,
                 m_utterances: int,
                 transform,
                 root_dir='/', 
                 ):
        super().__init__(source, root_dir) 
        self.m_utterances = m_utterances
        self.transform = transform
        self.paths = pd.DataFrame([], columns=['dataset', 'speaker', 'path', 'speaker_id'])

        if 'LibriTTS' in repos:
            self._load_libritts(repos['LibriTTS'])
        
    def _load_libritts(self, info={}):
        # loads all utterances into data [ dataframe, speaker, path ]
        file_paths = self.source.member_paths('/LibriTTS/'+info['version'])
        data = []
        for path in file_paths:
            if path[-3:] == 'wav' and path.split('/')[-1][0] != '.':
                data.append([
                    'LibriTTS', 
                    path.split('/')[-1].split('_')[0], 
                    path
                ])
        self._add_to_paths(data)

    def _add_to_paths(self, data):
        # loads data into dataframe -- dataset | speaker | path | speaker_id
        existing_data = self.paths[['dataset', 'speaker', 'path']]
        data = pd.DataFrame(data, columns=existing_data.columns)
        new_data = pd.concat([existing_data, data]).drop_duplicates()
        # add unique speaker_id column
        unique = new_data[['dataset', 'speaker']].drop_duplicates().reset_index().reset_index()
        unique = unique.rename(columns={unique.columns[0]:'speaker_id'})[['dataset', 'speaker', 'speaker_id']]
        self.paths = pd.merge(new_data, unique, on=['dataset', 'speaker'], how='left')
        return len(self.paths)

    def __len__(self):
        return self.paths.speaker_id.max() + 1

    def __getitem__(self, idx):
        # m_ut, partials_in_longest, n_mels, frames
        paths = self.paths[self.paths.speaker_id == idx].path
        if len(paths.values) < self.m_utterances:
            m_paths = [ random.choice(list(paths.values)) for _ in range(self.m_utterances) ]
        else:
            m_paths = random.sample(list(paths.values), self.m_utterances)
        values = self._get_m_utterances(m_paths)
        values = self._run_transforms(values)
        labels = np.array([ idx for _ in range(self.m_utterances) ])
        return labels, values

    def _get_m_utterances(self, paths):
        values = []
        for path in paths:
            data = self.source.load(path)
            data, _ = librosa.load(data)
            values.append(data)
        return values

    def _run_transforms(self, values):
        for transform in self.transform:
            values = [
                transform(utterance)
                for utterance in values
            ]
        return np.array(values)

