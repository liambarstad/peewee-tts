import os
import math
import torch
import mlflow
import argparse
from datasets import TextAudioDataset
from models import Tacotron2
from torch import nn
from torch.utils.data import DataLoader
from transforms import collate
from transforms import transform
from utils import Params
from metrics.metrics import Metrics

parser = argparse.ArgumentParser(description='Trains the speaker recognition encoder, generating embeddings for different speakers')
parser.add_argument('--config-path', type=str, help='path to config .yml file')
parser.add_argument('--save-model', type=str, help='whether or not to save the model')
parser.set_defaults(save_model='False')

args = parser.parse_args().__dict__
save_model = args['save_model'] != 'False'

params = Params(args['config_path'])

if params.meta['mlflow_remote_tracking']:
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
#else:
#    mlflow.set_tracking_uri(f'file:{os.getcwd()}')

params.save()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
def to_ordinal(data):
    return np.array([ params.transforms['char_values'].index(d) for d in data ])

dataset = TextAudioDataset(
    source=params.train['source'],
    root_dir=params.train['root_dir'],
    repos=params.train['repos'],
    transform={
        'text': [
            #transform.OneHotEncodeCharacters(values=params.transforms['char_values'])
            to_ordinal,
        ],
        'audio': [
            transform.MelSpec(**params.mel)
        ]
    }
)

dataloader = DataLoader(
        dataset, 
        params.model['batch_size'], 
        shuffle=params.train['shuffle'],
        collate_fn=collate.MaxPad(axis=(0, 1))
)

model = Tacotron2(**params.model, char_values=params.transforms['char_values'])#.to(device)
model.train()

optimizer = torch.optim.Adam(
    model.parameters(),
    betas=(params.train['beta_0'], params.train['beta_1']), 
    eps=params.train['eps']
)

loss = nn.MSELoss(reduction='mean')

per_epoch = math.ceil(len(dataset) / params.model['batch_size'])
metrics = Metrics(params.train['epochs'], per_epoch)

current_step = 0

for epoch in range(params.train['epochs']):
    for i, (text, audio) in enumerate(dataloader):

        # teacher forcing takes audio as input
        predictions = model(text, audio)

        # decay to params.train['decay_iterations'] == 50000 

        outputs = loss(predictions, audio)

        optimizer.zero_grad()

        outputs.backward()

        optimizer.step()

        import ipdb; ipdb.sset_trace()
        '''
        with torch.no_grad():
            metrics.add_step({
                
            })

        '''

        current_step += 1

