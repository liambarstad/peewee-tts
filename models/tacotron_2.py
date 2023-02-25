import mlflow
import torch
from torch import nn

class Tacotron2(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 characters_window: int,
                 num_layers: list,
                 hidden_size: int,
                 batch_size: int
                 ):
        super(Tacotron2, self).__init__()

        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.characters_window = 5

        import math   
        self.conv1 = nn.Conv1d(
            in_channels=self.embedding_size, 
            out_channels=math.ceil(self.embedding_size / self.characters_window),
            kernel_size=5
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

    def forward(self, x):

        x = self.conv1(x)

        return x

    def save(self):
        mlflow.pytorch.log_model(self, 'model')
        mlflow.pytorch.log_state_dict(self.state_dict(), artifact_path='state_dict')

