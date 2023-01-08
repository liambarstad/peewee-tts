import os
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SpeakerAudioDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 sources: dict,
                 transforms=[]
                 ):
        
        self.root_dir = root_dir
        self.sources = sources
        self.transforms = transforms

        self.paths = pd.DataFrame([], columns=['dataset', 'speaker', 'path', 'speaker_id'])

        if 'LibriTTS' in sources:
            self._load_libritts(sources['LibriTTS'])

    def num_speakers(self):
        return len(self.paths.speaker_id.value_counts())
        
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

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        info = self.paths.iloc[idx]  
        data, _ = librosa.load(info.path)
        values = (info.speaker_id, data)
        if self.transform:
            values = self.transform(values)
        return values

