# copy from https://medium.com/@hasithsura/audio-classification-d37a82d6715
import os
import numpy as np
import pandas as pd
from PIL import Image
import librosa
# import torchlibrosa as tl

from config import CONFIG


def convert2sec(s):
    time = s.split(':')
    sec = 60 * float(time[0]) + float(time[1])
    return sec


def get_info():
    id_df = pd.read_csv(f'./annotation/{CONFIG.data.target}/youtube.csv', header=0)
    ids = id_df['ID'].tolist()

    start_times = id_df['start_time'].tolist()
    end_times = id_df['end_time'].tolist()
    start_secs = list(map(convert2sec, start_times))
    end_secs = list(map(convert2sec, end_times))

    lengths = [e - s for (s, e) in zip(start_secs, end_secs)]

    return ids, lengths


def augment_noise():
    pass  # TODO


def get_melspectrogram_db(file_path,
                          offset=0,
                          duration=None,
                          sr=None,
                          n_fft=2048,
                          hop_length=512,
                          n_mels=128,
                          fmin=20,
                          fmax=8300,
                          top_db=80):

    wav, sr = librosa.load(file_path, sr=sr, offset=offset, duration=duration)
    if wav.shape[0] < 5 * sr:
        wav = np.pad(wav,
                     int(np.ceil((5 * sr - wav.shape[0]) / 2)),
                     mode='reflect')
    else:
        wav = wav[:5 * sr]
    spec = librosa.feature.melspectrogram(wav,
                                          sr=sr,
                                          n_fft=n_fft,
                                          hop_length=hop_length,
                                          n_mels=n_mels,
                                          fmin=fmin,
                                          fmax=fmax)
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db


def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


def main() -> None:
    ids, lengths = get_info()

    for (id, length) in zip(ids, lengths):
        sound_file_path = f'../dataset/sounds/{id}.mp3'
        output_dir = f'../dataset/mfcc/{id}'
        os.makedirs(output_dir, exist_ok=True)

        # non-overlapping windows
        # NOTE: 本当に3秒だけでスキルを判定できる？要検討
        window_width = CONFIG.data.mfcc_window

        i = 0
        while i * window_width < length:
            file_idx = str(i).zfill(6)
            start_segment = i * window_width
            try:
                spec = get_melspectrogram_db(sound_file_path,
                                             offset=start_segment,
                                             duration=window_width)
                mfcc_arr = spec_to_image(spec)
                mfcc_img = Image.fromarray(mfcc_arr)
                mfcc_img.save(f'{output_dir}/{file_idx}.png')
            except Exception:
                pass
            finally:
                i += 1


if __name__ == "__main__":
    main()
