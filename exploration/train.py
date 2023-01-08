from datasets import SpeakerAudioDataset
from dataloaders import SpeakerAudioDataLoader
from models import SpeakerVerificationLSTMEncoder
from transforms import MelSpec, ClipShuffle
import transforms.transform_utils

sample_rate = 22050
mel_params = {
    'sample_rate': sample_rate,
    'n_fft': int(1024 * (sample_rate / 16000)),
    'hop_length': int(256 * (sample_rate / 16000)),
    'win_length': int(1024 * (sample_rate / 16000)),
    'n_mels': 80
}

train_params = {
    'N_speakers': 64,
    'M_utterances': 10,
    'sources': {
        'LibriTTS': {
            'version': 'dev-clean'
        }
    }
}

model_params = {
    'input_size': 80,
    'hidden_size': 257,
    'projection_size': 256,
    'embedding_size': 256,
    'num_layers': 3
}

batch_size = train_params['N_speakers'] / train_params['M_utterances']
