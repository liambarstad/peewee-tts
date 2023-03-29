import os
import mlflow
import math
import time
import torch
import argparse
from statistics import mean
from datasets import SpeakerAudioDataset
from torch.utils.data import DataLoader
from models import SpeakerVerificationLSTMEncoder
from transforms import transform
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
            transform.ReduceNoise(**params.noise_reduce),
            # voice activity detection
            transform.VAD(**params.vad),
            # convert to mel spectrogram
            transform.MelSpec(**params.mel),
            # split into clip with length t
            transform.ParsePartial(**params.clip),
        ]
    )

dataloader = DataLoader(
    dataset, 
    batch_size=params.train['N_speakers'], 
    shuffle=params.train['shuffle']
)

model = SpeakerVerificationLSTMEncoder(**params.model).to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), **params.optimizer)
mlflow.log_param('optimizer', 'Adam')

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    **params.lr_scheduler
)

mlflow.log_param('scheduler', 'CosineAnnealingWarmRestarts')

per_epoch = math.ceil(len(dataset) / params.train['N_speakers'])

metrics = Metrics(params.train['epochs'], per_epoch)

try:
    for epoch in range(params.train['epochs']):
        for i, (speakers, data) in enumerate(dataloader):
            t1 = time.time()

            predictions = model(data.to(device))

            softmax_loss, contrast_loss = model.criterion(predictions)
            loss = softmax_loss + contrast_loss
            optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), **params.grad_clip)
            optimizer.step()
            scheduler.step()

            t2 = time.time()

            with torch.no_grad():
                metrics.add_step({ 
                    'loss': loss.mean().item(),
                    'softmax_loss': softmax_loss.mean().item(),
                    'contrast_loss': contrast_loss.mean().item(),
                    'exec_time': t2 - t1,
                    'learning_rate': scheduler.get_last_lr()[0]
                })

        metrics.agg_epoch('loss', agg_fn=mean)
        metrics.agg_epoch('softmax_loss', agg_fn=mean)
        metrics.agg_epoch('contrast_loss', agg_fn=mean)
        metrics.agg_epoch('exec_time', agg_fn=sum)
        metrics.agg_epoch('learning_rate', agg_fn=lambda x: x[-1])
except KeyboardInterrupt:
    print('TRAINING LOOP TERMINATED BY USER')

metrics.save()
print('METRICS SAVED')

if save_model:
    model.save()
    print('MODEL SAVED')

