import math
import torch
import mlflow
import argparse
from datasets import TextAudioDataset
from models import Tacotron2
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from utils import Params

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

dataset = TextAudioDataset(
    source=params.train['source'],
    root_dir=params.train['root_dir'],
    repos=params.train['repos']
    transform=Compose([
    ])
)

dataloader = DataLoader(dataset, params.model['batch_size'], shuffle=params.train['shuffle'])

model = Tacotron2(**params.model)#.to(device)
model.train()

optimizer = torch.optim.Adam(
    model.parameters(),
    betas=(params.train['beta_0'], params.train['beta_1']), 
    eps=params.train['eps']
)

per_epoch = math.ceil(len(dataset) / params.model['batch_size'])
current_step = 0

for epoch in range(params.train['epochs']):
    for i, (text, audio) in enumerate(dataloader):
        # text should be character embeddings
        # batch_size | 
        current_step += 1

        # decay to params.train['decay_iterations'] == 50000 
        # batch_size = 64

        predictions = model(input_text.to(device))
   
        loss = output_audio 
        

