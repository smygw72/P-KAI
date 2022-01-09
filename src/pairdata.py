import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchaudio.transforms import TimeMasking

from src.utils import seed_worker
from src.audio import get_samples

class PairRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def id1(self):
        return self._data[0]

    @property
    def id2(self):
        return self._data[1]

    @property
    def label(self):
        return self._data[2]


class PairDataSet(Dataset):
    def __init__(self, cfg, train_or_test, split_id):
        self.cfg = cfg
        self.train_or_test = train_or_test
        self.pair_list = []
        self.split_id = split_id
        self._parse_list()

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, index):
        record = self.pair_list[index]

        if record.label == '1':
            sup_id = record.id1
            inf_id = record.id2
            label_sim = False
        elif record.label == '-1':
            sup_id = record.id2
            inf_id = record.id1
            label_sim = False
        elif record.label == '0':
            sup_id = record.id1
            inf_id = record.id2
            label_sim = True

        sup = self._sampling(sup_id)
        inf = self._sampling(inf_id)

        return sup, inf, label_sim

    def _parse_list(self):
        file_path = f'./annotation/{self.cfg.data.target}/{self.split_id}/{self.train_or_test}_pair.csv'
        for row in open(file_path):
            record = PairRecord(row.strip().split(','))
            if record.label != 'X':
                self.pair_list.append(record)


    def _sampling(self, sound_id):
        sound_path = f'../dataset/{sound_id}.mp3'
        samples = get_samples(self.cfg, sound_path)

        total_frame = len(samples)
        n_frame = self.cfg.learning.sampling.n_frame

        img_size = self.cfg.data.img_size
        raw_samples = torch.Tensor(n_frame, 1, img_size, img_size)
        img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        transformed_samples = torch.Tensor(n_frame, 1, img_size, img_size)

        if self.cfg.learning.sampling.method == 'sparse':
            segment_len = int(total_frame / n_frame)
            for i in range(n_frame):
                start_idx = i * segment_len
                end_idx = (i + 1) * segment_len - 1
                if i == (n_frame - 1):
                    end_idx = total_frame - 1
                idx = random.randint(start_idx, end_idx)
                raw_samples = samples[idx]
                augmented_samples = self.augment(raw_samples)
                transformed_samples[i] = img_transform(augmented_samples)
        elif self.cfg.learning.sampling.method == 'dense':
            start_idx = random.randint(0, total_frame - n_frame)
            for i in range(n_frame):
                raw_samples = samples[start_idx + i]
                augmented_samples = self.augment(raw_samples)
                transformed_samples[i] = img_transform(augmented_samples)

        return transformed_samples

    def augment(self, sample):
        if self.train_or_test == 'train':
            if self.cfg.learning.augmentation.time_masking is True:
                time_masking = TimeMasking(time_mask_param=80)
                sample = time_masking(sample)
        return sample


def get_dataloader(cfg, train_or_test, split_id):

    if train_or_test == 'train':
        dataset = PairDataSet(cfg, 'train', split_id)
        batch_size = cfg.learning.train.batch_size
        shuffle = True
    else:
        dataset = PairDataSet(cfg, 'test', split_id)
        batch_size = cfg.learning.test.batch_size
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.learning.n_worker,
        worker_init_fn=seed_worker,
        pin_memory=True,
        drop_last=True
    )
    return dataloader
