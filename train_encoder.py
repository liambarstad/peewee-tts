import math
from datasets import SpeakerAudioDataset
from torch.utils.data import DataLoader
from models import SpeakerVerificationLSTMEncoder
from transforms import MelSpec, ClipShuffle
from transforms.transform_utils import ToTensor
from torchvision.transforms import Compose

sample_rate = 22050
window_len_ms = 25
hop_len_ms = 10

mel_params = {
    'sample_rate': sample_rate,
    'hop_length_ms': 10,
    'win_length_ms': 25,
    'n_mels': 80
}

clip_params = {
    't': 160
}

train_params = {
    'N_speakers': 6,
    'M_utterances': 10,
    'root_dir': 'data/utterance_corpuses',
    'sources': {
        'LibriTTS': {
            'version': 'dev-clean'
        }
    },
}

model_params = {
    'input_size': 80,
    'hidden_size': 257,
    'projection_size': 256,
    'embedding_size': 256,
    'num_layers': 3
}

dataset = SpeakerAudioDataset(
        root_dir=train_params['root_dir'],
        sources=train_params['sources'],
        m_utterances=train_params['M_utterances'],
        transform=Compose([
            # convert to mel spectrogram
            MelSpec(**mel_params),
            # split into clips with length t
            ClipShuffle(**clip_params),
        ])
    )

dataloader = DataLoader(dataset, batch_size=train_params['N_speakers'], shuffle=True) 

epochs = 1
model = SpeakerVerificationLSTMEncoder(**model_params)

import torch

for epoch in range(epochs):

    #cks = torch.Tensor()

    for i, (speakers, data) in enumerate(dataloader):

        # needs to take average

        num_samples = train_params['N_speakers'] * train_params['M_utterances']

        batch = data.reshape(
                num_samples,
                clip_params['t'],
                mel_params['n_mels']
                )

        labels = speakers.reshape(num_samples)

        predictions = model(batch.float())

        # does the order matter for the samples in the same batch?
        # is reshaping t and n_mels correct?

        import ipdb; ipdb.set_trace()

        # get batch_size examples
        # forward/backwards + update
        # if i % 5 == 0 (ex):
        #     print/visualize/progress
        #     e.g. epoch, step, input_size, graph, convergence

        break

