import main as inference

import _paths
from config import CONFIG


def main():
    train_id_path = f'{CONFIG.common.annotation_dir}/{CONFIG.learning.split_id}/train_id.csv'
    test_id_path = f'{CONFIG.common.annotation_dir}/{CONFIG.learning.split_id}/test_id.csv'

    scores = {}
    for row in open(train_id_path):
        sound_id = row.rstrip()
        print(sound_id)
        sound_path = f'{CONFIG.common.sound_dir}/{sound_id}.mp3'
        scores[sound_id] = inference.main(sound_path)

    for row in open(test_id_path):
        sound_id = row.rstrip()
        print(sound_id)
        sound_path = f'{CONFIG.common.sound_dir}/{sound_id}.mp3'
        scores[sound_id] = inference.main(sound_path)

    score_sorted = sorted(scores.items(), key=lambda x: x[1])
    print(score_sorted)


if __name__ == "__main__":
    main()
