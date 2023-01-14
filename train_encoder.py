import mlflow
import math
import torch
import argparse
from datasets import SpeakerAudioDataset
from torch.utils.data import DataLoader
from models import SpeakerVerificationLSTMEncoder
from transforms import MelSpec, ClipShuffle
from transforms.transform_utils import ToTensor
from torchvision.transforms import Compose
from utils import Params

parser = argparse.ArgumentParser(description='Trains the speaker recognition encoder, generating embeddings for different speakers')
parser.add_argument('--config_path', type=str, help='path to config .yml file')
args = parser.parse_args().__dict__

params = Params(args['config_path'])

dataset = SpeakerAudioDataset(
        root_dir=params.train['root_dir'],
        sources=params.train['sources'],
        m_utterances=params.train['M_utterances'],
        transform=Compose([
            # convert to mel spectrogram
            MelSpec(**params.mel),
            # split into clips with length t
            ClipShuffle(**params.clip),
        ])
    )

dataloader = DataLoader(dataset, batch_size=params.train['N_speakers'], shuffle=params.train['shuffle']) 

model = SpeakerVerificationLSTMEncoder(**params.model)


for epoch in range(params.train['epochs']):

    #cks = torch.Tensor()

    for i, (speakers, data) in enumerate(dataloader):

        # needs to take average

        num_samples = params.train['N_speakers'] * params.train['M_utterances']

        batch = data.reshape(
                num_samples,
                params.clip['t'],
                params.mel['n_mels']
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

