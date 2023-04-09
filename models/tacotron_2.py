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
                 attn_hidden_size: int,
                 attn_inverse_temperature: float,
                 attn_window_length: int,
                 attn_n_filters: int,
                 attn_conv_kernel_size: int,
                 batch_size: int,
                 speaker_embedding_model_uri: str,
                 char_values: str,
                 device='cpu'
                 ):
        super(Tacotron2, self).__init__()

        self.embedding_size = embedding_size
        self.conv_layers, _, _, _ = num_layers
        self.batch_size = batch_size
        self.device = device

        self.character_embedding = nn.Embedding(len(char_values), self.embedding_size)

        self.speaker_embedding_model = mlflow.pytorch.load_model(model_uri=speaker_embedding_model_uri).to(device)

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    self.embedding_size, 
                    self.embedding_size,
                    kernel_size=encoder_conv_kernel_size,
                    padding=int((encoder_conv_kernel_size - 1) / 2) 
                ), nn.BatchNorm1d(self.embedding_size)
            ) for _ in range(self.conv_layers)
        ])

        self.bidirectional_lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            bidirectional=True
        )

        self.location_sensitive_attention = LocationSensitiveAttention(
            hidden_dims=attn_hidden_size,
            embedding_dims=self.embedding_size,
            inverse_temperature=attn_inverse_temperature,
            window_length=attn_window_length,
            n_filters=attn_n_filters,
            conv_kernel_size=attn_conv_kernel_size,
        )
        
    def forward(self, text, audio):
        x = self.character_embedding(text.int())
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        x, _ = self.bidirectional_lstm(x)
        
        if self.speaker_embedding_model.training:
            self.speaker_embedding_model.eval()
        speaker_embedding = self.speaker_embedding_model(audio)
        import ipdb; ipdb.sset_trace()

        if self.training:
            pass
            # teacher forcing on ground truth
        else:
            pass
            # turn inference on


        '''
        speaker_embedding = self.speaker_embedding_model
        if not ground_truth:
            pass
        else:
            prenet_output = self.prenet(x)
            seq = self.location_sensitive_attention(x)
        '''


        return x

    def save(self):
        mlflow.pytorch.log_model(self, 'model')
        mlflow.pytorch.log_state_dict(self.state_dict(), artifact_path='state_dict')

