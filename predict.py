import librosa
import torch
from params import Params
from transforms import transform
from utils import load_model

config = Params('config/predict.yml')
melspec = transform.MelSpec(**config.mel)
sp_encoder = load_model(config.models['speaker_embedding'])
text_ordinal = transform.ToOrdinal(config.train['char_values'])
synthesizer = load_model(config.models['tt2'])
vocoder = transform.InverseMelSpec(**config.inverse_mel)

def predict(text, sp_file):
    audio_data, _ = librosa.load(sp_file)
    # from sources
    audio_mels = torch.tensor(melspec(audio_data)).unsqueeze(0)
    sp_embeddings = sp_encoder(audio_mels)
    #sp_embeddings = torch.zeros(1, 1)

    text_input = torch.tensor(text_ordinal(text)).unsqueeze(0)
    mels = synthesizer(text_input, sp_embeddings)

    return mels
    #mels = mels.squeeze(0).transpose(0, 1).numpy()
    #audio_output = vocoder(mels)
    #return audio_output