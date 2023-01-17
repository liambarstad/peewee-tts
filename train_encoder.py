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

parser = argparse.ArgumentParser(description='Trains the speaker recognition encoder, generating embeddings for different speakers')
parser.add_argument('--config_path', type=str, help='path to config .yml file')
args = parser.parse_args().__dict__

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

model = SpeakerVerificationLSTMEncoder(**params.model)
speaker_centroids = SpeakerCentroids()

optimizer = torch.optim.Adam(model.parameters(), lr=params.train['learning_rate'])

for epoch in range(params.train['epochs']):
    for i, (speakers, data) in enumerate(dataloader):

        # 80, 160
        # [ 80 ] ...160
        # [ 160 ] ...80
        # transpose

        import ipdb; ipdb.sset_trace()
        batch = data.reshape(-1, params.clip['t'], params.mel['n_mels']).to(device)

        labels = speakers.reshape(batch.shape[0]).to(device)

        predictions = model(batch.float())

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
    
        j_centroids = j_centroids.reshape(-1, 1, j_centroids.shape[1])
        predictions = predictions.reshape(-1, 1, predictions.shape[1])

        loss = model.criterion(predictions, j_centroids, k_centroids)
        optimizer.zero_grad()
        loss.backward()

        # TODO: decrease by half at every 30M steps
        optimizer.step()
            
        print(f'epoch: {epoch+1}/{params.train["epochs"]}, step: {i+1}, loss: {loss.item()}')

        #     e.g. epoch, step, input_size, graph, convergence

