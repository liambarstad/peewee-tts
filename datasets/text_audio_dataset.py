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
                 num_samples=None
                ):
        super().__init__(source, root_dir)
        self.transform = transform
        self.paths = pd.DataFrame([], columns=['dataset', 'speaker', 'text_path', 'audio_path'])

        if 'LibriTTS' in repos:
            self._load_libritts(repos['LibriTTS'])
        
        if num_samples:
            self.paths = self.paths.sample(n=num_samples)
            for p in self.paths.audio_path.values:
                print(p)

    def _load_libritts(self, info={}):
        file_paths = pd.Series(self.source.member_paths('/LibriTTS/'+info['version']))
        wavs = file_paths[file_paths.str.endswith('.wav')].str[:-4]
        txts = file_paths[file_paths.str.endswith('.normalized.txt')].str[:-15]
        f_roots = pd.Series(list(set(wavs) & set(txts)))
        paths_to_add = pd.DataFrame({
            'dataset': ['LibriTTS']*len(f_roots),
            'speaker': f_roots.str.split('/').str[-1].str.split('_').str[0],
            'text_path': f_roots+'.normalized.txt',
            'audio_path': f_roots+'.wav'
        })
        paths_to_add = paths_to_add[paths_to_add.speaker != '.']
        self.paths = pd.concat([self.paths, paths_to_add]).reset_index().drop(columns='index')

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

