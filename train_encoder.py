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
from utils import Params, SpeakerCentroids
from metrics.metrics import Metrics
from metrics.encoder_metrics import contrast_metric, loss_metric

parser = argparse.ArgumentParser(description='Trains the speaker recognition encoder, generating embeddings for different speakers')
parser.add_argument('--config-path', type=str, help='path to config .yml file')
parser.add_argument('--save-model', type=str, help='whether or not to save the model')
parser.set_defaults(save_model='False')

args = parser.parse_args().__dict__
save_model = args['save_model'] != 'False'

params = Params(args['config_path'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

'''
class GreaterThanZeroConstraint(object):
    def __init__(self):
        pass

    def __call__(self,module):
        if hasattr(module,'weight'):
            import ipdb; ipdb.sset_trace()
            w=module.weight.data
            w=w.clamp(0.5,0.7)
          module.weight.data=w
'''

model = SpeakerVerificationLSTMEncoder(**params.model).to(device)
#model._modules['W'].apply(GreaterThanZeroConstraint())
speaker_centroids = SpeakerCentroids()

optimizer = torch.optim.Adam(model.parameters(), lr=params.train['learning_rate'])

total_steps = math.ceil(len(dataset) / params.train['N_speakers']) * params.train['epochs']

metrics = Metrics(total_steps, model, save_model=save_model)

metrics.add_counter('contrast', contrast_metric, 5)
metrics.add_counter('loss', loss_metric, 5)

for epoch in range(params.train['epochs']):
    for i, (speakers, data) in enumerate(dataloader):

        batch = data.reshape(-1, params.clip['t'], params.mel['n_mels']).to(device)

        labels = speakers.reshape(batch.shape[0]).to(device)

        predictions = model(batch.float())

        if i == 0 and epoch == 0:
            metrics.add_graph(batch)

        speakers_data = speakers.numpy()
        predictions_data = predictions.detach().numpy()

        # update centroids of data based on mean of predictions, eji
        for speaker_id in speakers_data[:, 0]:
            preds_for_speaker = predictions_data[labels == speaker_id]
            eji = np.mean(preds_for_speaker, axis=0)
            speaker_centroids.append_data(speaker_id, eji)
        
        # get the centroids for utterances from the same speaker j
        j_centroids = np.array([ speaker_centroids.get_for_speaker(l) for l in labels.numpy() ])

        # get the centroids for all other speakers k
        k_centroids = np.array([
            [ j_centroids[(j + i) % len(j_centroids)] for i in range(len(speakers) - 1) ]
            for j in range(len(j_centroids)) 
        ])

        # forward pass
        loss, cos_similarity_j, cos_similarity_k = model.criterion(predictions, j_centroids, k_centroids)
        optimizer.zero_grad()

        loss.sum().backward()

        # TODO: decrease by half at every 30M steps
        optimizer.step()

        curr_step = i + (epoch * int(total_steps / params.train['epochs']))

        metrics.calculate(
            curr_step,
            loss=loss,
            predictions=predictions, 
            cos_similarity_j=cos_similarity_j,
            cos_similarity_k=cos_similarity_k
        )

metrics.save()

# save model
