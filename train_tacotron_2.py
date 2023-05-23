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
from params import Params
from transforms import collate
from transforms import transform
from utils import load_model, create_mask_matrix, generate_stop_token_labels
from metrics.metrics import Metrics

parser = argparse.ArgumentParser(description='Trains Tacotron 2, utilizing a saved model to generate a speaker encoding')
parser.add_argument('--config-path', type=str, help='path to config .yml file')
parser.add_argument('--save-model', type=str, help='whether or not to save the model')
parser.set_defaults(save_model='False')

args = parser.parse_args().__dict__
config_path = args['config_path']
save_model = args['save_model']

params = Params(config_path, save_model=save_model)

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

if params.meta['mlflow_remote_tracking']:
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
#else:
#    mlflow.set_tracking_uri(f'file:{os.getcwd()}')

params.save()

device = torch.device('cpu')
if 'cuda_gpu' in params.meta:
    device = torch.device(f'cuda:{params.meta["cuda_gpu"]}')
    assert device.type == 'cuda'

dataset = TextAudioDataset(
    source=params.train['source'],
    root_dir=params.train['root_dir'],
    repos=params.train['repos'],
    transform={
        'text': [
            transform.ToOrdinal(params.transforms['char_values']),
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
    collate_fn=collate.MaxPad(labels_axis=0, values_axis=0, min_val=1e-8),
    num_workers=params.train['num_workers']
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
    lr=params.train['lr_max'],
    betas=(params.train['beta_0'], params.train['beta_1']), 
    eps=params.train['eps'],
    weight_decay=params.train['l2_reg_weight']
)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=params.train['scheduler_gamma']
)

per_epoch = math.ceil(len(dataset) / params.model['batch_size'])
metrics = Metrics(params.train['epochs'], per_epoch)

iteration_count = 0

sp_lstm_encoder = load_model(params.speaker_embedding_model['model_uri'], 'cpu')

try:
    for epoch in range(params.train['epochs']):
        for i, (text, audio) in enumerate(dataloader):

            optimizer.zero_grad()
            # teacher forcing takes audio as input
            mask_matrix = create_mask_matrix(audio)
            stop_token_mask = (torch.sum(mask_matrix, dim=2) > 0)
            speaker_embeddings = sp_lstm_encoder(audio)
            #speaker_embeddings = torch.zeros(audio.shape[0], 1)

            print(audio.shape[1])
            before_postnet_preds, after_postnet_preds, stop_token_preds = model(
                text.to(device), 
                speaker_embeddings.to(device),
                audio.to(device)
            )

            stop_token_labels = generate_stop_token_labels(mask_matrix, params.train['stop_token_threshold'])

            before_postnet_preds.masked_fill_(mask_matrix, 0.)
            after_postnet_preds.masked_fill_(mask_matrix, 0.)
            stop_token_preds.masked_fill_(stop_token_mask, params.train['stop_token_threshold'])

            before_postnet_loss = nn.MSELoss()(before_postnet_preds, audio.to(torch.float32))
            after_postnet_loss = nn.MSELoss()(after_postnet_preds, audio.to(torch.float32))
            after_postnet_l1 = nn.L1Loss()(after_postnet_preds, audio.to(torch.float32))
            stop_token_loss = nn.BCELoss()(stop_token_preds, stop_token_labels)

            (before_postnet_loss + after_postnet_loss + after_postnet_l1 + stop_token_loss).backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), params.train['grad_clip_norm']
            )
            # convert to phonemes?

            optimizer.step()

            with torch.no_grad():
                correct_sts_neg = torch.sum((stop_token_preds < 0.5) * stop_token_mask.logical_not()).item()\
                    / torch.sum(stop_token_labels == 0.).item()

                correct_sts_pos = torch.sum((stop_token_preds > 0.5) * ~stop_token_mask.logical_not()).item()\
                    / torch.sum(stop_token_labels == 1.).item()
                    
                stop_diff = 0 
                for i, sample in enumerate(stop_token_labels):
                    stop_ind = torch.nonzero(sample == 1., as_tuple=False)[0].item()
                    stop_val = stop_token_preds[i, stop_ind].item()
                    stop_diff += stop_val - 1.

                metrics.add_step({
                    'before_loss': before_postnet_loss.item(),
                    'after_loss': after_postnet_loss.item(),
                    'after_l1_loss': after_postnet_l1.item(),
                    'stop_token_loss': stop_token_loss.item(),
                    'correct_sts_neg': correct_sts_neg,
                    'correct_sts_pos': correct_sts_pos,
                    'stop_val_diff': stop_diff / stop_token_preds.shape[0]
                }, round_num=4)

            #from utils import debug_memory; debug_memory()
            iteration_count += 1
        
        if iteration_count >= params.train['decay_iterations'] * (64 / params.model['batch_size']):
            scheduler.step()
            print(f'LR: {str(scheduler.get_last_lr())}')

        metrics.agg_epoch('before_loss', mean)
        metrics.agg_epoch('after_loss', mean)
        metrics.agg_epoch('after_l1_loss', mean)
        metrics.agg_epoch('stop_token_loss', mean)
        metrics.agg_epoch('correct_sts_neg', mean)
        metrics.agg_epoch('correct_sts_pos', mean)
        metrics.agg_epoch('stop_val_diff', mean)

except KeyboardInterrupt:
    print('TRAINING LOOP TERMINATED BY USER') 

metrics.save()

if params.save_model:
    model.save()