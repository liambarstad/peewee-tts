import sys
import librosa
import numpy as np
sys.path.append('.')
from transforms import transform, collate

test_wav, _ = librosa.load('test/test.wav', sr=22050)

def test_to_ordinal():
    to_ordinal = transform.ToOrdinal(char_values='xyz ')

    assert np.array_equal(to_ordinal('xy'), np.array([1, 2]))
    assert np.array_equal(to_ordinal('XY'), np.array([1, 2]))
    assert np.array_equal(to_ordinal(' Xy'), np.array([4, 1, 2]))
    assert np.array_equal(to_ordinal('ABcX'), np.array([1]))
    assert np.array_equal(to_ordinal('xyzxyzxyz '), to_ordinal('xyzxyzxyz '))

def test_reduce_noise():
    reduce_noise = transform.ReduceNoise(
        sample_rate=22050,
        prop_decrease=3.0,
        n_fft=512
    )
    reduced = reduce_noise(test_wav)
    assert (reduced != test_wav).sum() > 0
    assert len(reduced) == len(test_wav)

def test_mel_spec():
    mel_spec = transform.MelSpec(
        sample_rate=22050,
        win_length_ms=50,
        hop_length_ms=12.5,
        n_mels=80 
    )
    spec = mel_spec(test_wav)
    assert spec.shape[1] == 80

def test_inverse_mel_spec():
    mel_params = {
        'sample_rate': 22050,
        'win_length_ms': 50,
        'hop_length_ms': 12.5
    }
    mel_spec = transform.MelSpec(**mel_params, n_mels=80)
    inv_mel_spec = transform.InverseMelSpec(**mel_params)
    mels = mel_spec(test_wav)
    inverted = inv_mel_spec(mels.transpose(1, 0))

    assert len(inverted) < len(test_wav) + 100 or len(inverted) > len(test_wav) - 100