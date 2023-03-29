import mlflow
import torch
from torch import nn
from torch.nn import functional as func
from .constraints import GreaterThan0Constraint, GradientScaleParams

class SpeakerVerificationLSTMEncoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 projection_size: int,
                 embedding_size: int,
                 batch_size: int,
                 proj_scale_factor: float,
                 w_scale_factor: float,
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
        self.w_scale_factor = w_scale_factor

        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            proj_size=self.projection_size,
            batch_first=True
        )
        self._grad_scale_lstm_proj()

        self.linear = nn.Linear(
            in_features=self.projection_size,
            out_features=self.embedding_size
        )

        self.relu = nn.ReLU()

        self.W = nn.Parameter(torch.Tensor([10]))
        self._constrain_W_parameter()

        self.B = nn.Parameter(torch.Tensor([-5]))

        self.j_i_centroids = None
        self.j_k_centroids = None 

        self._init_weights()

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
        with torch.no_grad():
            self.j_i_centroids = (torch.sum(output, dim=1) / (output.shape[1] - 1)).reshape(x.shape[0], 1, -1)
            self.j_k_centroids = torch.mean(output, dim=1).reshape(x.shape[0], 1, -1)

        return output

    def criterion(self, predictions):
        cs = nn.CosineSimilarity(dim=2)
        cos_similarity_j = cs(predictions, self.j_i_centroids)
        sji = self.W * cos_similarity_j + self.B

        k_centroids = []
        for i, _ in enumerate(self.j_k_centroids):
            all_others = torch.cat((self.j_k_centroids[:i], self.j_k_centroids[i+1:]))
            k_centroids.append(all_others.view(-1, all_others.shape[-1]))
        k_centroids = torch.stack(tuple(k_centroids))
        # shape: N_speakers, N_speakers - 1, Embed Size

        csk = nn.CosineSimilarity(dim=1)
        sjk = torch.stack(tuple([
            torch.stack(tuple([
                self.W * csk(utterance.reshape(1, -1), k_centroids[i]) + self.B
                for utterance in p
            ]))
            for i, p in enumerate(predictions)
        ]))

        softmax_loss = torch.log(torch.sum(torch.exp(sjk), dim=2)) - sji
        contrast_loss = 1 - torch.sigmoid(sji) + torch.argmax(torch.sigmoid(sjk), dim=2) 

        return softmax_loss, contrast_loss

    def save(self):
        mlflow.pytorch.log_model(self, 'model')
        mlflow.pytorch.log_state_dict(self.state_dict(), artifact_path='state_dict')

    def _grad_scale_lstm_proj(self):
        lstm_weight_names = self.lstm._flat_weights_names
        proj_layer_names = [ 
            name for name in lstm_weight_names 
            if name.startswith('weight_hr') 
        ]
        gc = GradientScaleParams(proj_layer_names, self.proj_scale_factor)
        self.lstm.register_full_backward_hook(gc)

    def _constrain_W_parameter(self):
        wcp = GradientScaleParams(['W'], self.w_scale_factor)
        self.register_full_backward_hook(wcp)
        wg0 = GreaterThan0Constraint(['W'])
        self.register_full_backward_hook(wg0)

    def _init_weights(self):
        weight_fn = nn.init.orthogonal_
        for w in self.lstm._all_weights:
            for param in w:
                if 'weight' in param:
                    weight_fn(self.lstm.__getattr__(param), gain=3)
        weight_fn(self.linear.weight, gain=3)

