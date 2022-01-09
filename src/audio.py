import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T

torchaudio.set_audio_backend('sox_io')  # linux/macos
# torchaudio.set_audio_backend('soundfile')  # windows

def get_mfccs(cfg, filepath):
    waveform, __ = torchaudio.load(filepath)

    n_fft = 2048
    hop_length = 512
    n_mels = 128
    n_mfcc = 128  # 縦軸の解像度

    sample_rate = 44100
    to_mfcc = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': n_fft,
            'n_mels': n_mels,
            'hop_length': hop_length
        }
    )
    to_mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    if cfg.data.feature == 'mel_spectrogram':
        mfcc = to_mel_spectrogram(waveform[0])  # (n_mel, time)
    elif cfg.data.feature == 'mfcc':
        mfcc = to_mfcc(waveform[0])  # (n_mfcc, time)


    # segmentation
    time_len = cfg.data.time_len
    mfcc_len = int(sample_rate * time_len / hop_length)
    mfccs = torch.split(mfcc, mfcc_len, dim=1)
    if mfccs[-1].shape[1] != mfcc_len:  # drop last element if needs
        mfccs = list(mfccs)[:-1]

    return torch.stack(mfccs, dim=0)  # (N, n_mfcc, mfcc_len)
