import os
import math
import argparse
import torch
import mlflow
from statistics import mean
from datasets import TextAudioDataset
from models.tacotron_2 import Tacotron2
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
config_path = args['config_path']
save_model = args['save_model']

params = Params(config_path, save_model=save_model)

if params.meta['mlflow_remote_tracking']:
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
#else:
#    mlflow.set_tracking_uri(f'file:{os.getcwd()}')

params.save()

device = torch.device('cpu')
if 'cuda_gpu' in params.meta:
    device = torch.device(f'cuda:{params.meta["cuda_gpu"]}')
    assert device.type == 'cuda'

import numpy as np
class ToOrdinal:
    def __call__(self, data):
        return np.array([
            params.transforms['char_values'].index(d) + 2
            if d in params.transforms['char_values'] else 1
            for d in data
        ])

dataset = TextAudioDataset(
    source=params.train['source'],
    root_dir=params.train['root_dir'],
    repos=params.train['repos'],
    transform={
        'text': [
            #transform.OneHotEncodeCharacters(values=params.transforms['char_values'])
            #to_ordinal,
            ToOrdinal(),
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
        collate_fn=collate.MaxPad(labels_axis=0, values_axis=0)
)

model = Tacotron2(
    **params.model, 
    n_mels=params.mel['n_mels'],
    char_values=params.transforms['char_values'],
    device=device
).to(device)

model.train()

optimizer = torch.optim.Adam(
    model.parameters(),
    betas=(params.train['beta_0'], params.train['beta_1']), 
    eps=params.train['eps']
)

criterion = nn.MSELoss(reduction='mean')

per_epoch = math.ceil(len(dataset) / params.model['batch_size'])
metrics = Metrics(params.train['epochs'], per_epoch)

current_step = 0

def create_mask_matrix(audio):
    mask_matrix = torch.rand(*audio.shape)#.to(audio.device)
    for i, speaker in enumerate(audio):
        for j, ts in enumerate(speaker):
            if ts.sum() == 0.:
                mask_matrix[i][j] = torch.ones(*ts.shape)
            else:
                mask_matrix[i][j] = torch.zeros(*ts.shape)
    return mask_matrix == 1.

def generate_stop_token_labels(mask_matrix, threshold):
    token_labels = torch.masked_fill(mask_matrix.float(), (mask_matrix > 0.0), params.train['stop_token_threshold'])
    end_col = torch.tensor([threshold])\
        .expand(token_labels.shape[0], 1, token_labels.shape[-1])\
        .to(token_labels.device)
    token_labels = torch.cat((token_labels, end_col), dim=1).mean(dim=2)
    return token_labels[:, 1:]

from models.speaker_verification_lstm_encoder import SpeakerVerificationLSTMEncoder
sp_lstm_params = Params(params.speaker_embedding_model['config_path'], save_model=False)
sp_lstm_encoder = SpeakerVerificationLSTMEncoder(**sp_lstm_params.model)
state_dict = mlflow.pytorch.load_state_dict(params.speaker_embedding_model['state_dict_uri'])
sp_lstm_encoder.load_state_dict(state_dict)
sp_lstm_encoder.eval()
#encoder = mlflow.pytorch.load_model(model_uri=params.model['speaker_embedding_model_uri'])

import gc
memory_dict = []
all_objs = []

def debug_memory():
    ind = len(memory_dict)
    memory_dict.append([])
    all_objs.append([])
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                memory_dict[ind].append([type(obj), obj.size()])
            else:
                all_objs[ind].append([type(obj), obj.size()])
        except:
            pass
    import ipdb; ipdb.sset_trace()

try:
    for epoch in range(params.train['epochs']):
        for i, (text, audio) in enumerate(dataloader):

            optimizer.zero_grad()
            # teacher forcing takes audio as input
            speaker_embeddings = sp_lstm_encoder(audio)
            #speaker_embeddings = torch.FloatTensor(
            #    params.model['batch_size'], 
            #    params.model['speaker_embedding_size'],
            #).uniform_(0., 1.).to(device)
            #audio = audio.to(device)
            mel_predictions, stop_token_predictions = model(text.to(device), audio.to(device), speaker_embeddings.to(device))

            # run successfully
            # refactor / weight init
            # inference
            # wavenet
            # convert to phonemes?

            mask_matrix = create_mask_matrix(audio)
            # mask losses with padded values
            prediction_loss = criterion(mel_predictions[:, :-1, :], audio[:, 1:, :].to(torch.float32))
            prediction_loss = torch.masked_fill(prediction_loss, mask_matrix, 0.0)

            stop_token_labels = generate_stop_token_labels(mask_matrix, params.train['stop_token_threshold'])
            stop_token_loss = criterion(stop_token_predictions, stop_token_labels)

            (prediction_loss.mean() + stop_token_loss).backward()

            # decay to params.train['decay_iterations'] == 50000 

            optimizer.step()

            with torch.no_grad():
                metrics.add_step({
                    'prediction_loss': prediction_loss.mean().item(),
                    'stop_token_loss': stop_token_loss.item() 
                })

            #debug_memory()
            current_step += 1

        metrics.agg_epoch('prediction_loss', mean)
        metrics.agg_epoch('stop_token_loss', mean)

except KeyboardInterrupt:
    print('TRAINING LOOP TERMINATED BY USER') 

if params.save_model:
    model.save()