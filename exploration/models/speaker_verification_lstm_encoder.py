import torch
from torch import nn
from torch.nn import functional as func

class SpeakerVerificationLSTMEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 projection_size,
                 embedding_size,
                 num_layers
                ):
        super(SpeakerVerificationLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers

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

    def forward(self, x):
        # (64, 636, 80)

        # lstm with projection
        _, (hx, cx) = self.lstm(x)

        # linear layer w/ relu
        x = self.relu(self.linear(hx[-1]))

        # l2 normalize
        x = func.normalize(x, p=2, dim=1)

        return x

