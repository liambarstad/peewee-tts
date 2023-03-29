import mlflow
import torch
from torch import nn
from torch.nn import functional as F
from .layers.location_sensitive_attention import LocationSensitiveAttention

class Tacotron2(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 num_layers: list,
                 encoder_conv_kernel_size: int,
                 lstm_hidden_size: int,
                 location_sensitive_hidden_size: int,
                 batch_size: int,
                 char_values: str
                 ):
        super(Tacotron2, self).__init__()

        self.embedding_size = embedding_size
        self.conv_layers, _, _, _ = num_layers
        self.batch_size = batch_size

        self.character_embedding = nn.Embedding(len(char_values), self.embedding_size)

        # conv = nn.Conv1d(170, math.floor(170 / 5), kernel_size=5)

        
        self.conv_layers = [
            nn.Sequential(
                nn.Conv1d(
                    self.embedding_size, 
                    self.embedding_size,
                    kernel_size=encoder_conv_kernel_size,
                    padding=int((encoder_conv_kernel_size - 1) / 2) 
                ), nn.BatchNorm1d(self.embedding_size)
            ) for _ in range(self.conv_layers)
        ]

        self.bidirectional_lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            bidirectional=True
        )

        self.location_sensitive_attention = LocationSensitiveAttention(
            hidden_dims=location_sensitive_hidden_size
        )
        
        # 5, 512
        # conv
        # embeddings 512
        # num_layers[0] conv layers of shape 5x1
        # batch normalization
        # relu activation

        '''
        which are passed through a stack of 3 convolutional layers each containing 512 filters with shape 5 × 1, i.e., where
each filter spans 5 characters, followed by batch normalization [18]
and ReLU activations. As in Tacotron, these convolutional layers
model longer-term context (e.g., N-grams) in the input character
sequence. The output of the final convolutional layer is passed into a
single bi-directional [19] LSTM [20] layer containing 512 units (256
in each direction) to generate the encoded features.
        '''

    def forward(self, text, ground_truth=None):
        x = self.character_embedding(text.int())
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        x, (_, _) = self.bidirectional_lstm(x)

        seq = self.location_sensitive_attention(x)

        if not ground_truth:
            # turn inference on
            pass
        else:
            pass
            # teacher forcing on ground truth

        return x

    def save(self):
        mlflow.pytorch.log_model(self, 'model')
        mlflow.pytorch.log_state_dict(self.state_dict(), artifact_path='state_dict')

