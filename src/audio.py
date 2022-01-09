import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T

torchaudio.set_audio_backend('sox_io')  # linux/macos
# torchaudio.set_audio_backend('soundfile')  # windows

def get_samples(cfg, filepath):
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
        samples = to_mel_spectrogram(waveform[0])  # (n_mel, time)
    elif cfg.data.feature == 'mfcc':
        samples = to_mfcc(waveform[0])  # (n_mfcc, time)

    # segmentation
    time_len = cfg.data.time_len
    frame_len = int(sample_rate * time_len / hop_length)
    splitted_samples = torch.split(samples, frame_len, dim=1)
    if splitted_samples[-1].shape[1] != frame_len:  # drop last element if needs
        splitted_samples = list(splitted_samples)[:-1]

    return torch.stack(splitted_samples, dim=0)
