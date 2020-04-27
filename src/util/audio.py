import librosa
import librosa.filters
import numpy as np
import hyperparams as hp
from scipy.io import wavfile
from scipy import signal
import lws

def load_wav(path):
    return librosa.core.load(path, sr=hp.sample_rate)[0]


def save_wav(wav, path):
    wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hp.sample_rate, wav.astype(np.int16))


def preemphasis(x, coef=0.85):
    b = np.array([1., -coef], x.dtype)
    a = np.array([1.], x.dtype)
    return signal.lfilter(b, a, x)


def inv_preemphasis(x, coef=0.85):
    b = np.array([1.], x.dtype)
    a = np.array([1., -coef], x.dtype)
    return signal.lfilter(b, a, x)



def spectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(np.abs(D)) - hp.ref_db
    return _normalize(S)


def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) + hp.ref_db)  # Convert back to linear
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** hp.sharpening_factor)
    y = processor.istft(D).astype(np.float32)
    return inv_preemphasis(y)


def melspectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_db
    if not hp.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - hp.min_db >= 0
    return _normalize(S)


def _lws_processor():
    return lws.lws(hp.n_fft, hp.hop_size, mode="speech")

# Conversions:


_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    if hp.fmax is not None:
        assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(hp.sample_rate, hp.n_fft,
                               fmin=hp.fmin, fmax=hp.fmax,
                               n_mels=hp.n_mels)


def _amp_to_db(x):
    min_level = np.exp(hp.min_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - hp.min_db) / -hp.min_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hp.min_db) + hp.min_db