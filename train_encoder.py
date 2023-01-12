import math
from datasets import SpeakerAudioDataset
from dataloaders import SpeakerAudioDataLoader
from models import SpeakerVerificationLSTMEncoder
from transforms import MelSpec, ClipShuffle
from transforms.transform_utils import ToTensor
from torchvision.transforms import Compose

sample_rate = 22050
window_len_ms = 25
hop_len_ms = 10

mel_params = {
    'sample_rate': sample_rate,
    'hop_length_ms': 10,
    'win_length_ms': 25,
    'n_mels': 80
}

clip_params = {
    'sample_rate': sample_rate,
    'win_length_ms': 25,
    'fixed_length_ms': 800,
}

train_params = {
    'N_speakers': 64,
    'M_utterances': 10,
    'root_dir': 'data/utterance_corpuses',
    'sources': {
        'LibriTTS': {
            'version': 'dev-clean'
        }
    },
}

model_params = {
    'input_size': 80,
    'hidden_size': 257,
    'projection_size': 256,
    'embedding_size': 256,
    'num_layers': 3
}

batch_size = train_params['N_speakers'] / train_params['M_utterances']

dataset = SpeakerAudioDataset(
        root_dir=train_params['root_dir'],
        sources=train_params['sources'],
        transform=Compose([
            ClipShuffle(**clip_params),
            MelSpec(**mel_params),
        ])
    )

#print(len(dataset))
#print(dataset.num_speakers())
#print(dataset[0])

dataloader = SpeakerAudioDataLoader(dataset, 
        batch_size=64, 
        m_utterances=train
        collate_fn=collate)



for i, (speaker, audio) in enumerate(dataloader):
    print(i, (speaker, audio))
    break

'''
dataloader = SpeakerAudioDataLoader(
    dataset, 
    train_params['N_speakers'], 
    train_params['M_utterances'], 
    800
)


# transforms ToTensor

epochs = 3
total_samples = len(dataset)
total_speakers = dataset.num_speakers()
n_iterations = math.ceil(total_samples / batch_size)

model = SpeakerVerificationLSTMEncoder(**model_params)

for epoch in range(epochs):
      
    cks = torch.Tensor(256,                                                                                                                                 
    [
        torch.init_some_stuff_here(embedding_dims)
        for speaker in total_speakers
    ]
                                                                                                                                       
    for i, (speaker, audio) in enumerate(dataloader):
        embed = model(audio)
                                                                                                                                       
        # get batch_size examples
        # forward/backwards + update
        # if i % 5 == 0 (ex):
        #     print/visualize/progress
        #     e.g. epoch, step, input_size, graph, convergence
        
'''
