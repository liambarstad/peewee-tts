import mlflow
import torch
import torch.nn.functional as F
from torch import nn
from .constraints import GreaterThan0Constraint, GradientScaleParams

class SpeakerVerificationLSTMEncoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 projection_size: int,
                 embedding_size: int,
                 batch_size: int,
                 proj_scale_factor: float,
                 wb_scale_factor: float,
                 num_layers: int
                ):
        super(SpeakerVerificationLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.proj_scale_factor = proj_scale_factor
        self.wb_scale_factor = wb_scale_factor

        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True
        )

        self.projection_layer = nn.Linear(
            self.hidden_size,
            self.projection_size
        )
        self._constrain_proj_layer()

        self.W = nn.Parameter(torch.Tensor([10.0]))
        self.B = nn.Parameter(torch.Tensor([-5.0]))
        self._constrain_WB_parameters()

        self.j_i_centroids = None
        self.j_k_centroids = None 

        self._init_weights()

    def forward(self, x):
        # constrain W to be > 0
        if self.W[0] <= 0:
            self.W = nn.Parameter(torch.Tensor([1e-6]))

        lstm_out, _ = self.lstm(x.reshape(-1, *x.shape[2:]).float())
        last_lstm = lstm_out[:, lstm_out.size(1)-1]
        projected = self.projection_layer(last_lstm.float())
        output = F.normalize(projected, p=2, dim=1)
        # reshape to N_speakers, M_utterances, Embed_size
        output = output.view(x.shape[0], x.shape[1], -1)

        # save speaker centroids for backwards pass
        with torch.no_grad():
            self.j_i_centroids = self._calculate_j_i_centroids(output)
            self.j_k_centroids = torch.mean(output, dim=1)

        return output

    def criterion(self, predictions):
        # positive component for similarity between utterance and its speaker's centroid (N, M)
        sji = self._calculate_sji(predictions)
        # negative component for similarity between utterance and other speakers' centroids (N, M, N-1)
        sjk = self._calculate_sjk(predictions)
        # softmax loss for TD-SV (equation 6) with difference between positive and negative component
        loss = torch.log(torch.sum(torch.exp(sjk), dim=2) + torch.exp(sji) + 1e-6) - sji
        return loss.sum()

    def save(self):
        mlflow.pytorch.log_model(self, 'model')
        mlflow.pytorch.log_state_dict(self.state_dict(), artifact_path='state_dict')

    def _constrain_proj_layer(self):
        plc = GradientScaleParams(['weight', 'bias'], self.proj_scale_factor)
        self.projection_layer.register_full_backward_hook(plc)

    def _constrain_WB_parameters(self):
        wcp = GradientScaleParams(['W', 'B'], self.wb_scale_factor)
        self.register_full_backward_hook(wcp)

    def _init_weights(self):
        for w in self.lstm._all_weights:
            for param in w:
                if 'weight' in param:
                    nn.init.orthogonal_(self.lstm.__getattr__(param), gain=3)
                elif 'bias' in param:
                    nn.init.constant_(self.lstm.__getattr__(param), 0.0)

    def _calculate_j_i_centroids(self, output):
        # get centroids of all the utterances for their speakers, excluding the utterance itself
        # returns N, M, embed_size
        centroids = torch.zeros(output.shape).to(output.device)
        for si, speaker in enumerate(output):
            for ui, _ in enumerate(speaker):
                other_utterances = torch.cat([speaker[:ui], speaker[ui+1:]])
                centroids[si][ui] = torch.mean(other_utterances, dim=0)
        return centroids
    
    def _calculate_sji(self, predictions):
        # returns the weighted/biased cosine similarity between each utterance and the centroid of its speaker (minus itself) 
        # returns N, M
        reshaped_preds = predictions.view(-1, predictions.shape[-1])
        reshaped_cents = self.j_i_centroids.reshape(-1, self.j_i_centroids.shape[-1])
        sji = self.W * (F.cosine_similarity(reshaped_preds, reshaped_cents) + 1e-6) + self.B
        return sji.view(*predictions.shape[:-1])

    def _calculate_sjk(self, predictions):
        # returns the weighted/biased cosine similarity between each utterance and the centroids of the other speakers
        # returns N, M, (N - 1)
        pred_view = predictions.view(*predictions.shape[:2], 1, predictions.shape[-1])
        pred_expanded = pred_view.expand(*predictions.shape[:2], self.j_k_centroids.shape[0] - 1, predictions.shape[-1])
        centroids = torch.zeros(*pred_expanded.shape).to(predictions.device)
        for si, sp in enumerate(centroids):
            for ui, _ in enumerate(sp):
                other_centroids = torch.cat([self.j_k_centroids[:si], self.j_k_centroids[si+1:]])
                centroids[si][ui] = other_centroids
        return self.W * (F.cosine_similarity(pred_expanded, centroids, dim=3) + 1e-6) + self.B
