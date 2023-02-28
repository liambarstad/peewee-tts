import librosa
import pandas as pd
from .dataset import Dataset

# https://arxiv.org/pdf/2010.10694v2.pdf

class TextAudioDataset(Dataset):
    def __init__(self,
                 source: str,
                 repos: dict,
                 root_dir='/',
                 transform={
                    'text': [],
                    'audio': []
                 },
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
        item = self.paths.iloc[idx]
        text_data = self._transform_text(
            self.source.load(item['text_path'])
        )
        audio_data = self._transform_audio(
            self.source.load(item['audio_path'])
        )
        return text_data, audio_data

    def _transform_text(self, bytesio):
        text = bytesio.getvalue().decode('utf-8')
        for transform in self.transform['text']:
            text = transform(text)
        return text

    def _transform_audio(self, bytesio):
        audio, _ = librosa.load(bytesio)
        for transform in self.transform['audio']:
            audio = transform(audio)
        return audio

