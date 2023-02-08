import mlflow
import torch
from torch import nn
from torch.nn import functional as func

class SpeakerVerificationLSTMEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 projection_size,
                 embedding_size,
                 batch_size,
                 num_layers
                ):
        super(SpeakerVerificationLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            proj_size=self.projection_size,
            batch_first=True
        )

        self.linear = nn.Linear(
            in_features=self.projection_size,
            out_features=self.embedding_size
        )

        self.relu = nn.ReLU()

        self.W = nn.Parameter(torch.Tensor([10]))
        # contstrain to be > 0

        self.B = nn.Parameter(torch.Tensor([5]))

        self.j_centroids = None 

    def forward(self, x):
        # lstm with projection
        _, (hx, cx) = self.lstm(x.reshape(-1, x.shape[2], x.shape[3]).float())

        # linear layer w/ relu
        output = self.relu(self.linear(hx[-1]))

        # l2 normalize
        output = func.normalize(output, p=2, dim=1)

        # reshape to N_speakers, M_utterances, Embed_size
        output = output.view(x.shape[0], x.shape[1], -1)

        # save speaker centroids for backwards pass
        self.j_centroids = torch.mean(output, dim=1).reshape(x.shape[0], 1, -1)

        return output.view(x.shape[0], x.shape[1], -1)

    def criterion(self, predictions):
        cs = nn.CosineSimilarity(dim=2)
        cos_similarity_j = cs(predictions, self.j_centroids)

        k_centroids = []
        for i, _ in enumerate(self.j_centroids):
            all_others = torch.cat((self.j_centroids[:i], self.j_centroids[i+1:]))
            k_centroids.append(all_others.view(-1, all_others.shape[-1]))

        # shape: N_speakers, N_speakers - 1, Embed Size
        k_centroids = torch.stack(tuple(k_centroids))

        softmax = nn.Softmax(dim=0)
        sji = softmax(self.W * cos_similarity_j + self.B)

        csk = nn.CosineSimilarity(dim=1)

        # shape: N_speakers, M_utterances, N_speakers - 1
        sjk = torch.stack(tuple([
            torch.stack(tuple([
                softmax(self.W + csk(utterance.reshape(1, -1), k_centroids[i]) + self.B) 
                for utterance in p
            ]))
            for i, p in enumerate(predictions)
        ]))

        loss = torch.log(torch.sum(torch.exp(sjk), dim=2)) - sji

        return loss, sji, sjk

    def save(self):
        mlflow.pytorch.log_model(self, 'model')
        mlflow.pytorch.log_state_dict(self.state_dict(), artifact_path='state_dict')


