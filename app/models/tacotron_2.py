import mlflow
import torch
from torch import nn
from torch.nn import functional as F
from .layers import LinearNorm, ConvNorm
from .location_sensitive_attention import LocationSensitiveAttention

class Prenet(nn.Module):
    def __init__(self, in_size, out_size, layers):
        super(Prenet, self).__init__()
        self.linear_layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)] + 
            [LinearNorm(out_size, out_size, bias=False)
             for _ in range(layers - 1)]
        )

    def forward(self, x):
        for lin in self.linear_layers:
            x = F.dropout(F.relu(lin(x)), p=0.5, training=True)
        return x

class Postnet(nn.Module):
    def __init__(self, input_dim, embedding_dim, kernel_size, num_layers):
        super(Postnet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])

        self.layers.append(
            nn.Sequential(
                ConvNorm(
                    in_channels=input_dim,
                    out_channels=embedding_dim,
                    kernel_size=kernel_size,
                    padding=int((kernel_size - 1) / 2),
                    w_init_gain='tanh'
                ), nn.BatchNorm1d(embedding_dim)
            )
        )

        for _ in range(1, num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    ConvNorm(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=kernel_size,
                        padding=int((kernel_size - 1) / 2),
                        w_init_gain='tanh'
                    ), nn.BatchNorm1d(embedding_dim)
                )
            )

        self.layers.append(
            nn.Sequential(
                ConvNorm(
                    in_channels=embedding_dim,
                    out_channels=input_dim,
                    kernel_size=kernel_size,
                    padding=int((kernel_size - 1) / 2),
                    w_init_gain='linear'
                ), nn.BatchNorm1d(input_dim)
            )
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.tanh(x)
            x = F.dropout(x, 0.5, self.training)
        return x

class Tacotron2(nn.Module):
    def __init__(self,
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

        self.character_embedding = nn.Embedding(len(char_values) + 1, self.embedding_size)

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                ConvNorm(
                    in_channels=self.embedding_size, 
                    out_channels=self.embedding_size,
                    kernel_size=self.encoder_conv_kernel_size,
                    padding=int((self.encoder_conv_kernel_size - 1) / 2),
                    w_init_gain='relu'
                ), nn.BatchNorm1d(self.embedding_size)
            ) for _ in range(self.num_conv_layers)
        ])

        self.bidirectional_lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
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

        self.decoder_projection = LinearNorm(
            self.decoder_lstm_hidden_size, self.n_mels, bias=True
        )

        self.stop_token_projection = LinearNorm(
            self.decoder_lstm_hidden_size + self.embedding_size + self.speaker_embedding_size, 
            1, bias=True, w_init_gain='sigmoid'
        )

        self.postnet = Postnet(
            input_dim=self.n_mels,
            embedding_dim=self.postnet_embedding_dim,
            kernel_size=self.postnet_kernel_size, 
            num_layers=self.num_postnet_conv_layers
        )
        
    def forward(self, text, speaker_embeddings, audio=None, **kwargs):
        text_embedding = self.encode(text)
        text_embedding = self._cat_with_voiceprint(text_embedding, speaker_embeddings)
        if self.training:
            assert audio != None
            return self.train_loop(text_embedding, audio)
        else:
            return self.inference(text_embedding, **kwargs)

    def encode(self, text):
        x = self.character_embedding(text.int())
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = F.dropout(x, 0.5, self.training)
        x = x.transpose(1, 2)
        self.bidirectional_lstm.flatten_parameters()
        x, _ = self.bidirectional_lstm(x)
        return x

    def train_loop(self, text_embedding, audio):
        # teacher forcing on ground truth
        before_postnet_preds = torch.zeros((audio.shape[0], audio.shape[1], self.n_mels))
        after_postnet_preds = torch.zeros((audio.shape[0], audio.shape[1], self.n_mels))
        stop_token_predictions = torch.zeros((audio.shape[0], audio.shape[1]))

        first_frame = torch.zeros(audio.shape[0], 1, audio.shape[2]).to(self.device)
        audio_input = torch.cat((first_frame, audio), dim=1)[:, :-1, :].to(torch.float32)
        prenet_output = self.prenet(audio_input)
        training_values = self._get_init_training_values(text_embedding)

        for time_step_ind in range(prenet_output.shape[1]):
            prenet_embeds = prenet_output[:, time_step_ind, :]
            decoder_output, training_values = self.decode_frame(prenet_embeds, text_embedding, training_values)

            attention_context = training_values[0]
            stop_token_prediction, mel_prediction, mel_prediction_residual = self.predict_frame(decoder_output, attention_context)

            stop_token_predictions[:, time_step_ind] = stop_token_prediction.squeeze(1)
            before_postnet_preds[:, time_step_ind, :] = mel_prediction 
            after_postnet_preds[:, time_step_ind, :] = mel_prediction + mel_prediction_residual.squeeze(2)

        return before_postnet_preds, after_postnet_preds, stop_token_predictions

    def inference(self, text_embedding, max_length=5000, threshold=0.5, **kwargs):
        with torch.no_grad():
            prediction_frames = torch.tensor([])
            residual_frames = torch.tensor([])
            stop_token = 0.0
            training_values = self._get_init_training_values(text_embedding)
            input_frame = torch.zeros(text_embedding.shape[0], 1, self.n_mels)
            while stop_token < threshold and (prediction_frames.shape[0] == 0 or prediction_frames.shape[1] < max_length):
                prenet_embeds = self.prenet(input_frame)
                decoder_output, training_values = self.decode_frame(prenet_embeds.squeeze(1), text_embedding, training_values)

                attention_context = training_values[0]
                stop_token_prediction, mel_prediction, mel_residual = self.predict_frame(decoder_output, attention_context) 

                stop_token = stop_token_prediction.item()
                input_frame = mel_prediction.unsqueeze(1)
                prediction_frames = torch.cat((prediction_frames, input_frame), dim=1)
                residual_frames = torch.cat((residual_frames, mel_residual.transpose(1, 2)), dim=1)
            return prediction_frames + residual_frames

    def decode_frame(self, input_frame, text_embedding, training_values):
        attention_context, enc_lstm_hidden, enc_lstm_cell, dec_lstm_hidden, dec_lstm_cell, attention_weights, attention_weights_cum = training_values
        enc_lstm_input = torch.cat((input_frame, attention_context), dim=1)
        enc_lstm_hidden, enc_lstm_cell = self.encoder_lstm(
            enc_lstm_input, 
            (enc_lstm_hidden, enc_lstm_cell)
        ) 
        enc_lstm_hidden = F.dropout(enc_lstm_hidden, 0.1, self.training)

        attention_context, attention_weights = self.location_sensitive_attention(
            enc_lstm_hidden, 
            text_embedding, 
            attention_weights,
            attention_weights_cum,
        )
        attention_weights_cum += attention_weights

        decoder_input = torch.cat((attention_context, enc_lstm_hidden), dim=1)
        dec_lstm_hidden, dec_lstm_cell = self.decoder_lstm(
            decoder_input,
            (dec_lstm_hidden, dec_lstm_cell)
        )
        dec_lstm_hidden = F.dropout(dec_lstm_hidden, 0.1, self.training)

        training_values = [attention_context, enc_lstm_hidden, enc_lstm_cell, dec_lstm_hidden, dec_lstm_cell, attention_weights, attention_weights_cum]
        return dec_lstm_hidden, training_values

    def predict_frame(self, decoder_output, attention_context):
        stop_token_input = torch.cat((decoder_output, attention_context), dim=1)
        stop_token_prediction = torch.sigmoid(self.stop_token_projection(stop_token_input))
        mel_prediction = self.decoder_projection(decoder_output)
        mel_prediction_residual = self.postnet(mel_prediction.unsqueeze(2))
        return stop_token_prediction, mel_prediction, mel_prediction_residual

    def _cat_with_voiceprint(self, text_embedding, speaker_embedding):
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

