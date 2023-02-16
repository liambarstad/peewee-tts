import io
import os
import random
import boto3
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .sources import AWSCloudSource, LocalDirectorySource

class SpeakerAudioDataset(Dataset):
    def __init__(self, 
                 source: str,
                 repos: dict,
                 m_utterances: int,
                 transform,
                 root_dir='/', 
                 load_from_cloud=False
                 ):
        
        self.root_dir = root_dir
        self.m_utterances = m_utterances
        self.transform = transform
        self.load_from_cloud = load_from_cloud

        self.paths = pd.DataFrame([], columns=['dataset', 'speaker', 'path', 'speaker_id'])

        if source == 'aws_cloud':
            self.source = AWSCloudSource(self.root_dir)
        elif source == 'local_directory':
            self.source = LocalDirectorySource(self.root_dir)

        if 'LibriTTS' in repos:
            self._load_libritts(repos['LibriTTS'])
        
    def _load_libritts(self, info={}):
        # loads all utterances into data [ dataframe, speaker, path ]
        file_paths = self.source.member_paths('/LibriTTS/'+info['version'])
        data = []
        for path in file_paths:
            if path[-3:] == 'wav':
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
        speaker_data = []
        if len(paths.values) < self.m_utterances:
            # drop data where the number of utterances < M
            speaker_data = [ np.zeros(1) for _ in range(self.m_utterances) ]
        else:
            # select m utterances at random
            m_paths = random.sample(list(paths.values), self.m_utterances)
            for path in m_paths:
                data = self.source.load(path)
                data, _ = librosa.load(data)
                speaker_data.append(data)
        values = self.transform(speaker_data)
        labels = np.array([ idx for _ in range(self.m_utterances) ])
        return labels, values


