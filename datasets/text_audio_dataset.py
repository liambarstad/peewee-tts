import pandas as pd
from .dataset import Dataset

class TextAudioDataset(Dataset):
    def __init__(self,
                 source: str,
                 repos: dict,
                 root_dir='/',
                 transform=None,
                ):
        super().__init__(source, root_dir)
        self.transform = transform
        self.paths = pd.DataFrame([], columns=['dataset', 'speaker', 'text_path', 'audio_path'])

        if 'LibriTTS' in repos:
            self._load_libritts(repos['LibriTTS'])

    def _load_libritts(self, info={}):
        file_paths = pd.Series(self.source.member_paths('/LibriTTS/'+info['version']))
        wavs = file_paths[file_paths.str.endswith('.wav')].str[:-4]
        txts = file_paths[file_paths.str.endswith('.normalized.txt')].str[:-15]
        f_roots = wavs[wavs.isin(txts)]
        for root in f_roots:
            self._add_path([
                'LibriTTS',
                root.split('/')[-1].split('_')[0],
                root+'.normalized.txt',
                root+'.wav'
            ])

    def _add_path(self, data):
        new_row = pd.DataFrame(
            [data], 
            columns=self.paths.columns, 
            index=[len(self.paths)]
        )
        self.paths = pd.concat([self.paths, new_row])

    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        pass
        
