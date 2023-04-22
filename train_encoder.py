import os
import mlflow
import math
import time
import torch
import argparse
from statistics import mean
from datasets import SpeakerAudioDataset
from torch.utils.data import DataLoader
from models.speaker_verification_lstm_encoder import SpeakerVerificationLSTMEncoder
from transforms import transform
from utils import Params
from metrics.metrics import Metrics

# https://arxiv.org/pdf/1710.10467.pdf
# Li Wan, Quan Wang, Alan Papir, and Ignacio Lopez Moreno, Generalized End-to-End Loss for Speaker Verification," Google Inc., USA

params = Params()

if params.meta['mlflow_remote_tracking']:
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

params.save()

device = torch.device('cpu')
if 'cuda_gpu' in params.meta:
    device = torch.device(f'cuda:{params.meta["cuda_gpu"]}')
    assert device.type == 'cuda'

dataset = SpeakerAudioDataset(
        root_dir=params.train['root_dir'] if 'root_dir' in params.train else '/',
        source=params.train['source'],
        repos=params.train['repos'],
        m_utterances=params.train['M_utterances'],
        transform=[
            # reduce noise for 'other' set
            # does not seem to increase performance, and takes a long time
            #transform.ReduceNoise(**params.noise_reduce),
            # voice activity detection
            #transform.VAD(**params.vad),
            # convert to mel spectrogram
            transform.MelSpec(**params.mel),
            # split into clip with length t
            transform.ParsePartial(**params.clip),
        ]
    )

dataloader = DataLoader(
    dataset, 
    batch_size=params.train['N_speakers'], 
    shuffle=params.train['shuffle'],
    num_workers=params.train['num_workers']
)

model = SpeakerVerificationLSTMEncoder(**params.model).to(device)
model.train()

optimizer = torch.optim.SGD(model.parameters(), **params.optimizer)
mlflow.log_param('optimizer', 'SGD')

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, 
#    **params.lr_scheduler
#)

#mlflow.log_param('scheduler', 'ReduceLROnPlateau')

per_epoch = math.ceil(len(dataset) / params.train['N_speakers'])

metrics = Metrics(params.train['epochs'], per_epoch)

try:
    for epoch in range(params.train['epochs']):
        for i, (speakers, data) in enumerate(dataloader):
            t1 = time.time()

            optimizer.zero_grad()
            predictions = model(data.to(device))
            loss = model.criterion(predictions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.lstm.parameters(), **params.grad_clip)
            torch.nn.utils.clip_grad_norm_([
                model.W, 
                model.B, 
                *model.projection_layer.parameters()
            ], 1.0)

            optimizer.step()

            t2 = time.time()

            with torch.no_grad():
                metrics.add_step({ 
                    'loss': loss.mean().item(),
                    'exec_time': t2 - t1
                })
        #scheduler.step(loss.mean())

        metrics.agg_epoch('loss', agg_fn=mean)
        metrics.agg_epoch('exec_time', agg_fn=sum)
except KeyboardInterrupt:
    print('TRAINING LOOP TERMINATED BY USER')

metrics.save()

if params.save_model:
    model.save()

