import torch
from torch import nn

class LocationSensitiveAttention(nn.Module):
    def __init__(self,
                 hidden_dims: int,
                 embedding_dims: int,
                 window_length: int,
                 inverse_temperature: float,
                 n_filters: int,
                 conv_kernel_size: int
                 ):
        super(LocationSensitiveAttention, self).__init__()
        # end of sequence token

        self.hidden_dims = hidden_dims
        self.embedding_dims = embedding_dims
        self.window_length = window_length
        self.inverse_temperature = inverse_temperature
        self.n_filters = n_filters
        self.conv_kernel_size = conv_kernel_size

        self.query_layer = nn.Linear(
            self.window_length * self.hidden_dims, 
            self.hidden_dims,
            bias=False
        )
        self.value_layer = nn.Linear(
            self.embedding_dims, 
            self.hidden_dims,
            bias=False
        )
        self.score_layer = nn.Linear(
            self.hidden_dims,
            1,
            bias=True
        )
        self.conv_attn = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=self.n_filters,
                kernel_size=self.conv_kernel_size
            ),
            nn.Linear(
                self.n_filters,
                self.hidden_dims,
                bias=False
            )
        )

        self.b = nn.Parameter(torch.Tensor())

        self._init_weights()

    def forward(self, x):

        import ipdb; ipdb.sset_trace() 
        scores = self.score_layer(torch.tanh(
            self.query_layer(x)
            + self.value_layer(x)
            + self.conv_attn()
            + self.b
        ))
        
        # a(i,j) = sig(B*e(i,j)) / sum(sig(B*e(i,j))
        pass

    def _init_weights(self):
        #All the weight matrices were initialized from a normal Gaussian distribution 
        # with its standard deviation set to 0.01. Recurrent weights were orthogonalized
        pass