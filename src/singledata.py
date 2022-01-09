import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchaudio.transforms import TimeMasking

from src.utils import seed_worker
from src.audio import get_mfccs


class SingleDataSet(Dataset):
    def __init__(self, cfg, sound_path, transform):
        self.mfccs = get_mfccs(cfg, sound_path)
        self.transform = transform

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, index):
        mfcc = self.mfccs[index]
        mfcc = self.transform(mfcc)
        return mfcc


def get_dataloader(cfg, sound_path, transform):

    dataset = SingleDataSet(cfg, sound_path, transform)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.inference.n_frame,
        shuffle=False,
        num_workers=cfg.inference.n_worker,
        worker_init_fn=seed_worker,
        pin_memory=True,
        drop_last=True
    )
    return dataloader
