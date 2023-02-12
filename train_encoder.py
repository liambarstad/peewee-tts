import os
import mlflow
import math
import torch
import argparse
import numpy as np
from datasets import SpeakerAudioDataset
from torch.utils.data import DataLoader
from models import SpeakerVerificationLSTMEncoder
from transforms import MelSpec, ClipShuffle
from transforms.transform_utils import ToTensor
from torchvision.transforms import Compose
from utils import Params
from metrics.metrics import Metrics
from metrics.encoder_metrics import contrast_metric, loss_metric

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
        sources=params.train['sources'],
        m_utterances=params.train['M_utterances'],
        transform=Compose([
            # convert to mel spectrogram
            MelSpec(**params.mel),
            # split into clips with length t
            ClipShuffle(**params.clip),
        ]),
        load_from_cloud=params.meta['load_from_cloud']
    )

dataloader = DataLoader(dataset, batch_size=params.train['N_speakers'], shuffle=params.train['shuffle']) 

model = SpeakerVerificationLSTMEncoder(**params.model).to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=params.train['learning_rate'])

total_steps = math.ceil(len(dataset) / params.train['N_speakers']) * params.train['epochs']

metrics = Metrics(total_steps)

metrics.add_counter('contrast', contrast_metric, inc=5)
metrics.add_counter('loss', loss_metric, inc=5)

for epoch in range(params.train['epochs']):
    for i, (speakers, data) in enumerate(dataloader):

        predictions = model(data)

        # forward pass
        loss, sji, sjk = model.criterion(predictions)

        optimizer.zero_grad()

        # backward pass
        loss.sum().backward()

        # TODO: decrease lr by half at every 30M steps
        optimizer.step()

        curr_step = i + (epoch * int(total_steps / params.train['epochs'])) + 1

        metrics.calculate(
            # calculates each metric added, if curr_step % inc == 0, prints current step
            epoch+1,
            curr_step,
            loss=loss,
            sji=sji,
            sjk=sjk
        )

metrics.save()
print('METRICS SAVED')

if save_model:
    model.save()
    print('MODEL SAVED')

