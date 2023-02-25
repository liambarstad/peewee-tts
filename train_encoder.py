import os
import mlflow
import math
import torch
import argparse
import numpy as np
from statistics import mean
from datasets import SpeakerAudioDataset
from torch.utils.data import DataLoader
from models import SpeakerVerificationLSTMEncoder
from transforms import MelSpec, ClipShuffle
from torchvision.transforms import Compose
from utils import Params
from metrics.metrics import Metrics

# https://arxiv.org/pdf/1710.10467.pdf
# Li Wan, Quan Wang, Alan Papir, and Ignacio Lopez Moreno, Generalized End-to-End Loss for Speaker Verification," Google Inc., USA

parser = argparse.ArgumentParser(description='Trains the speaker recognition encoder, generating embeddings for different speakers')
parser.add_argument('--config-path', type=str, help='path to config .yml file')
parser.add_argument('--save-model', type=str, help='whether or not to save the model')
parser.set_defaults(save_model='False')

args = parser.parse_args().__dict__
save_model = args['save_model'] != 'False'

params = Params(args['config_path'])

if params.meta['mlflow_remote_tracking']:
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

params.save()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = SpeakerAudioDataset(
        root_dir=params.train['root_dir'] if 'root_dir' in params.train else '/',
        source=params.train['source'],
        repos=params.train['repos'],
        m_utterances=params.train['M_utterances'],
        transform=Compose([
            # convert to mel spectrogram
            MelSpec(**params.mel),
            # split into clips with length t
            ClipShuffle(**params.clip),
        ])
    )

dataloader = DataLoader(dataset, batch_size=params.train['N_speakers'], shuffle=params.train['shuffle']) 

model = SpeakerVerificationLSTMEncoder(**params.model).to(device)
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=params.train['learning_rate'])

per_epoch = math.ceil(len(dataset) / params.train['N_speakers'])

metrics = Metrics(params.train['epochs'], per_epoch)

for epoch in range(params.train['epochs']):
    for i, (speakers, data) in enumerate(dataloader):

        predictions = model(data.to(device))
        # need to send input to device, will send error if not

        # forward pass
        softmax_loss, contrast_loss = model.criterion(predictions)
        loss = softmax_loss + contrast_loss

        optimizer.zero_grad()

        # backward pass
        loss.mean().backward()

        # TODO: decrease lr by half at every 30M steps
        optimizer.step()

        with torch.no_grad():
            metrics.add_step({ 
                'loss': loss.mean().item(),
                'softmax_loss': softmax_loss.mean().item(),
                'contrast_loss': contrast_loss.mean().item(),
            })

    metrics.agg_epoch('loss', agg_fn=mean)
    metrics.agg_epoch('softmax_loss', agg_fn=mean)
    metrics.agg_epoch('contrast_loss', agg_fn=mean)

metrics.save()
print('METRICS SAVED')

if save_model:
    model.save()
    print('MODEL SAVED')

