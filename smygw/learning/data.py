import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from preprocess import get_melspectrogram_db, spec_to_image
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
        return len(self.data)

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

        sup = preprocess(sup_id)
        inf = preprocess(inf_id)

        return (sup, inf, similar)

    def _parse_list(self):
        file_path = f'./smygw/annotation/{CONFIG.dataset.split_id}/{self.train_or_test}.csv'
        self.pair_list = [
            PairRecord(x.strip().split(' ')) for x in open(file_path)
        ]


def sampling(id):
    pass


def get_dataloader(train_or_test):

    if train_or_test == 'train':
        dataset = PairDataSet('train')
        shuffle = True
    else:
        dataset = PairDataSet('test')
        shuffle = False

    data_loader = DataLoader(
        dataset,
        batch_size=CONFIG.learning.batch_size,
        shuffle=shuffle,
        num_workers=CONFIG.learning.n_worker,
        pin_memory=True
    )
    return data_loader
