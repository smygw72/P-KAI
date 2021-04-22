import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import CONFIG


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
            similar = False
        elif record.label == '-1':
            sup_id = record.id2
            inf_id = record.id1
            similar = False
        elif record.label == '0':
            sup_id = record.id1
            inf_id = record.id2
            similar = True

        sup = sampling(sup_id)
        inf = sampling(inf_id)

        return sup, inf, similar

    def _parse_list(self):
        file_path = f'{CONFIG.path.annotation_dir}/{CONFIG.split_id}/{self.train_or_test}_pair.csv'
        self.pair_list = [
            PairRecord(x.strip().split(',')) for x in open(file_path)
        ]


def sampling(id):
    files = os.listdir(f'{CONFIG.path.mfcc_dir}/{id}/')
    n_file = len(files)
    n_sample = CONFIG.n_sample
    segment_len = int(n_file / n_sample)

    mfcc_tensor = torch.Tensor(
        n_sample, 1, CONFIG.input_size, CONFIG.input_size
    )

    for i in range(n_sample):
        start_idx = i * segment_len
        end_idx = (i + 1) * segment_len
        if end_idx > n_file:
            end_idx = n_file
        idx = random.randint(start_idx, end_idx + 1)
        path = f'{CONFIG.path.mfcc_dir}/{id}/{files[idx]}'
        mfcc_tensor[i] = get_img(path)

    return mfcc_tensor


def get_img(path):
    transform = transforms.Compose([
        transforms.Resize((CONFIG.input_size, CONFIG.input_size)),
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

    data_loader = DataLoader(
        dataset,
        batch_size=CONFIG.batch_size,
        shuffle=shuffle,
        num_workers=CONFIG.n_worker,
        pin_memory=True
    )
    return data_loader
