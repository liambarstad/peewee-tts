import mlflow
import torch
from torch import nn
from torch.nn import functional as F
from .layers.location_sensitive_attention import LocationSensitiveAttention

class Prenet(nn.Module):
    def __init__(self, in_size, out_size, layers):
        super(Prenet, self).__init__()
        self.layers = layers
        self.linear_layers = nn.ModuleList(
            [nn.Linear(in_size, out_size, bias=False)] + 
            [nn.Linear(out_size, out_size, bias=False)
             for _ in range(self.layers - 1)]
        )
        self._init_weights()

    def forward(self, x):
        for lin in self.linear_layers:
            x = F.dropout(F.relu(lin(x)), p=0.5, training=True)
        return x

    def _init_weights(self):
        for layer in self.linear_layers:
             torch.nn.init.xavier_uniform_(
                layer.weight,
                gain=torch.nn.init.calculate_gain('linear')
            )

class Postnet(nn.Module):
    def __init__(self, input_dim, embedding_dim, kernel_size, num_layers):
        super(Postnet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])

        self.layers.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=embedding_dim,
                    kernel_size=kernel_size,
                    padding=int((kernel_size - 1) / 2)
                ), nn.BatchNorm1d(embedding_dim)
            )
        )

        for _ in range(1, num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=kernel_size,
                        padding=int((kernel_size - 1) / 2)
                    ), nn.BatchNorm1d(embedding_dim)
                )
            )

        self.layers.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=input_dim,
                    kernel_size=kernel_size,
                    padding=int((kernel_size - 1) / 2)
                ), nn.BatchNorm1d(input_dim)
            )
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.tanh(x)
            x = F.dropout(x, p=0.5, training=True)
        return x

class Tacotron2(nn.Module):
    def __init__(self,
                 speaker_embedding_model_uri: str,
                 embedding_size: int,
                 speaker_embedding_size: int,
                 num_layers: list,
                 encoder_conv_kernel_size: int,
                 lstm_hidden_size: int,
                 prenet_hidden_size: int,
                 encoder_lstm_hidden_size: int,
                 location_hidden_dim: int,
                 location_n_filters: int,
                 location_kernel_size: int,
                 attn_inverse_temperature: float,
                 decoder_lstm_hidden_size: int,
                 postnet_embedding_dim: int,
                 postnet_kernel_size: int,
                 batch_size: int,
                 n_mels: int,
                 char_values: str,
                 device='cpu'
                 ):
        super(Tacotron2, self).__init__()

        #self.speaker_embedding_model = mlflow.pytorch.load_model(model_uri=speaker_embedding_model_uri).to(device)

        self.embedding_size = embedding_size
        self.speaker_embedding_size = speaker_embedding_size
        self.num_conv_layers, self.num_prenet_layers, self.num_postnet_conv_layers = num_layers
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.prenet_hidden_size = prenet_hidden_size
        self.encoder_lstm_hidden_size = encoder_lstm_hidden_size
        self.location_hidden_dim = location_hidden_dim
        self.location_n_filters = location_n_filters
        self.location_kernel_size = location_kernel_size
        self.attn_inverse_temperature = attn_inverse_temperature
        self.decoder_lstm_hidden_size = decoder_lstm_hidden_size
        self.postnet_embedding_dim = postnet_embedding_dim
        self.postnet_kernel_size = postnet_kernel_size

        self.batch_size = batch_size
        self.n_mels = n_mels
        self.device = device

        self.character_embedding = nn.Embedding(len(char_values) + 2, self.embedding_size)

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=self.embedding_size, 
                    out_channels=self.embedding_size,
                    kernel_size=self.encoder_conv_kernel_size,
                    padding=int((self.encoder_conv_kernel_size - 1) / 2) 
                ), nn.BatchNorm1d(self.embedding_size)
            ) for _ in range(self.num_conv_layers)
        ])

        self.bidirectional_lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            bidirectional=True
        )

        self.prenet = Prenet(self.n_mels, self.prenet_hidden_size, self.num_prenet_layers)

        self.encoder_lstm = nn.LSTMCell(
            self.prenet_hidden_size + self.embedding_size + self.speaker_embedding_size,
            self.encoder_lstm_hidden_size
        )

        self.location_sensitive_attention = LocationSensitiveAttention(
            input_dim=self.encoder_lstm_hidden_size,
            embedding_dim=self.embedding_size + self.speaker_embedding_size,
            attention_dim=self.location_hidden_dim,
            location_n_filters=self.location_n_filters,
            location_kernel_size=self.location_kernel_size,
            inverse_temperature=self.attn_inverse_temperature
        )

        self.decoder_lstm = nn.LSTMCell(
            self.embedding_size + self.speaker_embedding_size + self.decoder_lstm_hidden_size,
            self.decoder_lstm_hidden_size
        )

        self.decoder_projection = nn.Linear(
            self.decoder_lstm_hidden_size, self.n_mels, bias=True
        )

        self.stop_token_projection = nn.Linear(
            self.decoder_lstm_hidden_size, 1, bias=True
        )

        self.postnet = Postnet(
            input_dim=self.n_mels,
            embedding_dim=self.postnet_embedding_dim,
            kernel_size=self.postnet_kernel_size, 
            num_layers=self.num_postnet_conv_layers
        )
        
    def forward(self, text, audio, speaker_embeddings):
        text_embedding = self.encode(text)
        text_embedding = self._cat_with_voiceprint(text_embedding, speaker_embeddings)
        
        if self.training:
            # teacher forcing on ground truth
            output_mels = torch.zeros((audio.shape[0], audio.shape[1], self.n_mels))
            stop_token_predictions = torch.zeros((audio.shape[0], audio.shape[1]))

            prenet_output = self.prenet(audio.float())
            attention_context, enc_lstm_hidden, enc_lstm_cell, dec_lstm_hidden, dec_lstm_cell, attention_weights, attention_weights_cum = self._get_init_training_values(text_embedding)

            for time_step_ind in range(prenet_output.shape[1]):
                input_frame = prenet_output[:, time_step_ind, :]
                enc_lstm_input = torch.cat((input_frame, attention_context), dim=1)
                enc_lstm_hidden, enc_lstm_cell = self.encoder_lstm(
                    enc_lstm_input, 
                    (enc_lstm_hidden, enc_lstm_cell)
                ) 
                # self.attention_hidden = F.dropout(
                # self.attention_hidden, self.p_attention_dropout, self.training)
                attention_context, attention_weights = self.location_sensitive_attention(
                    enc_lstm_hidden, 
                    text_embedding, 
                    attention_weights,
                    attention_weights_cum 
                )
                attention_weights_cum += attention_weights

                decoder_input = torch.cat((attention_context, enc_lstm_hidden), dim=1)

                dec_lstm_hidden, dec_lstm_cell = self.decoder_lstm(
                    decoder_input,
                    (dec_lstm_hidden, dec_lstm_cell)
                )

                stop_token_prediction = self.stop_token_projection(dec_lstm_hidden)
                stop_token_predictions[:, time_step_ind] = stop_token_prediction.squeeze(1)

                mel_prediction = self.decoder_projection(dec_lstm_hidden)
                mel_prediction_residual = self.postnet(mel_prediction.unsqueeze(2))
                output_mels[:, time_step_ind, :] = mel_prediction + mel_prediction_residual.squeeze(2)

            return output_mels, stop_token_predictions
        else:
            pass
            # turn inference on

    def encode(self, text):
        x = self.character_embedding(text.int())
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        x, _ = self.bidirectional_lstm(x)
        return x

    def _cat_with_voiceprint(self, text_embedding, speaker_embedding):
        #if self.speaker_embedding_model.training:
        #    self.speaker_embedding_model.eval()#.to('cpu')
        #import ipdb; ipdb.set_trace()
        #speaker_embedding = self.speaker_embedding_model(audio)#.to('cpu')).to(self.device)
        sp_embed_expanded = speaker_embedding.unsqueeze(1)\
            .expand(speaker_embedding.shape[0], text_embedding.shape[1], speaker_embedding.shape[-1])
        return torch.cat((sp_embed_expanded, text_embedding), dim=2)

    def _get_init_training_values(self, text_embedding):
        attention_context = torch.zeros(text_embedding.shape[0], text_embedding.shape[-1]).to(self.device)
        enc_lstm_hidden = torch.zeros(text_embedding.shape[0], self.encoder_lstm_hidden_size).to(self.device)
        enc_lstm_cell = torch.zeros(text_embedding.shape[0], self.encoder_lstm_hidden_size).to(self.device)
        dec_lstm_hidden = torch.zeros(text_embedding.shape[0], self.decoder_lstm_hidden_size).to(self.device)
        dec_lstm_cell = torch.zeros(text_embedding.shape[0], self.decoder_lstm_hidden_size).to(self.device)
        attention_weights = torch.zeros(text_embedding.shape[0], text_embedding.shape[1]).to(self.device)
        attention_weights_cum = torch.zeros(text_embedding.shape[0], text_embedding.shape[1]).to(self.device)

        return attention_context, enc_lstm_hidden, enc_lstm_cell, dec_lstm_hidden, dec_lstm_cell, attention_weights, attention_weights_cum
        
    def save(self):
        mlflow.pytorch.log_model(self, 'model')
        mlflow.pytorch.log_state_dict(self.state_dict(), artifact_path='state_dict')
        print('MODEL SAVED')

