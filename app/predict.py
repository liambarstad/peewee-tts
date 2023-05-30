import librosa
import torch
import transform

melspec = transform.MelSpec(**config.mel)
text_ordinal = transform.ToOrdinal(config.train['char_values'])
vocoder = transform.InverseMelSpec(**config.inverse_mel)

def get_speaker_embedding(sp_encoder, sp_files):
    speaker_embeddings = []
    for sp_file in sp_files:
        audio_data, _ = librosa.load(sp_file)
        audio_mels = torch.tensor(melspec(audio_data)).unsqueeze(0)
        speaker_embeddings.append(sp_encoder(audio_mels))
    return sum(speaker_embeddings) / len(speaker_embeddings)

def predict(text, sp_embeds, synthesizer):
    text_input = torch.tensor(text_ordinal(text)).unsqueeze(0)
    mels = synthesizer(text_input, sp_embeds)
    mels = mels.squeeze(0).transpose(0, 1).numpy()
    return vocoder(mels)