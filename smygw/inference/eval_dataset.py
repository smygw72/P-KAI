import main as inference

import _paths
from config import CONFIG


def main():
    train_id_path = f'{CONFIG.common.path.annotation_dir}/{CONFIG.learning.split_id}/train_id.csv'
    test_id_path = f'{CONFIG.common.path.annotation_dir}/{CONFIG.learning.split_id}/test_id.csv'

    for row in open(train_id_path):
        print(row)
        train_path = f'{CONFIG.common.path.sound_dir}/{row}.mp3'
        inference.main(train_path)

    for row in open(test_id_path):
        print(row)
        test_path = f'{CONFIG.common.path.sound_dir}/{row}.mp3'
        inference.main(test_path)


if __name__ == "__main__":
    main()
