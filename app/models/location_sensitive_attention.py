import torch
from torch import nn
from torch.nn import functional as F
from .layers import LinearNorm, ConvNorm

class LocationSensitiveAttention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 embedding_dim: int,
                 attention_dim: int,
                 location_n_filters: int,
                 location_kernel_size: int,
                 inverse_temperature: int
                 ):
        super(LocationSensitiveAttention, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.location_n_filters = location_n_filters
        self.location_kernel_size = location_kernel_size
        self.inverse_temperature = inverse_temperature

        self.query_layer = LinearNorm(
            self.input_dim,
            self.attention_dim,
            bias=False,
            w_init_gain='tanh'
        )

        self.value_layer = LinearNorm(
            self.embedding_dim,
            self.attention_dim,
            bias=False,
            w_init_gain='tanh'
        )

        self.score_layer = LinearNorm(
            self.attention_dim, 1, bias=False
        )

        self.location_conv = ConvNorm(
            in_channels=2,
            out_channels=self.location_n_filters,
            kernel_size=self.location_kernel_size,
            padding=int((self.location_kernel_size - 1) / 2),
            bias=False
        )

        self.location_linear = LinearNorm(
            self.location_n_filters,
            self.attention_dim,
            bias=False,
            w_init_gain='tanh'
        )

        #self.b = nn.Parameter(torch.rand(self.attention_dim).uniform_(-0.1, 0.1))

    def forward(self, query, values, attn_weights, attn_cum):
        location_sensitive_input = torch.cat((attn_weights.unsqueeze(1), attn_cum.unsqueeze(1)), dim=1)
        conv_output = self.location_conv(location_sensitive_input)
        location_alignment = self.location_linear(conv_output.transpose(1, 2))

        scores = self.score_layer(torch.tanh(
            self.query_layer(query.unsqueeze(1))
            + self.value_layer(values)
            + location_alignment
            #+ self.b
        )).squeeze(2)

        attn = F.softmax(self.inverse_temperature*scores, dim=1)
        #attn = torch.sigmoid(self.inverse_temperature*scores) / torch.sigmoid(self.inverse_temperature*torch.sum(scores, dim=0))+1e-8
        context = torch.bmm(attn.unsqueeze(dim=1), values).squeeze(dim=1)

        return context, attn