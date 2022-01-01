import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.utils import seed_worker
from config.config import CONFIG


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
    def __init__(self, train_or_test):
        self.train_or_test = train_or_test
        self.pair_list = []
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

        sup = sampling(sup_id)
        inf = sampling(inf_id)

        return sup, inf, label_sim

    def _parse_list(self):
        file_path = f'./annotation/{CONFIG.data.target}/{CONFIG.learning.split_id}/{self.train_or_test}_pair.csv'
        for row in open(file_path):
            record = PairRecord(row.strip().split(','))
            if record.label != 'X':
                self.pair_list.append(record)


def sampling(sound_id):
    sound_dir = f'../dataset/mfcc/{sound_id}/'
    files = os.listdir(sound_dir)
    n_file = len(files)
    n_frame = CONFIG.learning.sampling.n_frame
    segment_len = int(n_file / n_frame)

    mfcc_tensor = torch.Tensor(
        n_frame, 1, CONFIG.data.img_size, CONFIG.data.img_size
    )

    if CONFIG.learning.sampling.method == 'sparse':
        for i in range(n_frame):
            start_idx = i * segment_len
            end_idx = (i + 1) * segment_len - 1
            if i == (n_frame - 1):
                end_idx = n_file - 1
            idx = random.randint(start_idx, end_idx)
            path = f'{sound_dir}/{files[idx]}'
            mfcc_tensor[i] = get_img(path)
    elif CONFIG.learning.sampling.method == 'dense':
        start_idx = random.randint(0, n_file - n_frame)
        for i in range(n_frame):
            idx = start_idx + i
            path = f'{sound_dir}/{files[idx]}'
            mfcc_tensor[i] = get_img(path)

    return mfcc_tensor


def get_img(path):
    transform = transforms.Compose([
        transforms.Resize(
            (CONFIG.data.img_size, CONFIG.data.img_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path)
    img_tensor = transform(img)

    return img_tensor


def get_dataloader(train_or_test):

    if train_or_test == 'train':
        dataset = PairDataSet('train')
        shuffle = True
    else:
        dataset = PairDataSet('test')
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG.learning.batch_size,
        shuffle=shuffle,
        num_workers=CONFIG.learning.n_worker,
        worker_init_fn=seed_worker,
        pin_memory=True
    )
    return dataloader

