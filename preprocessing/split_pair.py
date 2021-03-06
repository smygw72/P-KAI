import os
import random
import csv
import os
import pandas as pd
import sys

sys.path.append('./src')
from utils import set_seed
sys.path.append('./config')
from config import get_config


def get_datalist(target):
    filepath = f'./annotation/{target}/datalist.txt'
    with open(filepath, 'r') as f:
        datalist = f.readlines()
    for i in range(len(datalist)):
        datalist[i] = datalist[i].replace('\n', '')
    random.shuffle(datalist)
    return datalist


def get_pairs(target):
    pair_df = pd.read_csv(f'./annotation/{target}/all_pair.csv', header=0)
    id1s = pair_df['id1'].tolist()
    id2s = pair_df['id2'].tolist()
    labels = pair_df['label'].tolist()
    return id1s, id2s, labels


def main(*args, **kwargs):
    cfg = get_config()
    target = cfg.data.target
    k = cfg.learning.k_fold

    set_seed(cfg.seed)

    all_ids = get_datalist(target)
    id1s, id2s, labels = get_pairs(target)

    subset_size = int(len(all_ids) / k)

    for i in range(k):
        output_dir = f'./annotation/{target}/{i}'
        os.makedirs(f'{output_dir}', exist_ok=True)

        # split id
        train_ids = []
        test_ids = []
        for j in range(k):
            start_idx = j * subset_size
            end_idx = (j + 1)*subset_size

            # Exception processing for last subset
            if end_idx > len(all_ids):
                end_idx = len(all_ids)

            # one of k-splitted list
            subset_ids = all_ids[start_idx:end_idx]

            if j != i:
                train_ids.extend(subset_ids)
            else:
                test_ids.extend(subset_ids)

        train_ids = sorted(train_ids)
        test_ids = sorted(test_ids)

        with open(f'{output_dir}/train_id.csv', mode='w', newline='') as f:
            writer = csv.writer(f)
            for train_id in train_ids:
                writer.writerow([train_id])

        with open(f'{output_dir}/test_id.csv', mode='w', newline='') as f:
            writer = csv.writer(f)
            for test_id in test_ids:
                writer.writerow([test_id])

        # split all_pair
        train_pairs = []
        test_pairs = []
        for id1, id2, label in zip(id1s, id2s, labels):

            if label == 'X':
                print(f"Warning: Not annotated for ({id1}, {id2})")

            d = {'id1': id1, 'id2': id2, 'label': label}

            # NOTE: (train_id, test_id) pairs are removed
            if (id1 in train_ids) and (id2 in train_ids):
                train_pairs.append(d)
            elif (id1 in test_ids) and (id2 in test_ids):
                test_pairs.append(d)

        with open(f'{output_dir}/train_pair.csv', mode='w', newline='') as f:
            writer = csv.writer(f)
            for pair in train_pairs:
                writer.writerow([pair["id1"], pair["id2"], pair["label"]])

        with open(f'{output_dir}/test_pair.csv', mode='w', newline='') as f:
            writer = csv.writer(f)
            for pair in test_pairs:
                writer.writerow([pair["id1"], pair["id2"], pair["label"]])


if __name__ == "__main__":
    main()
