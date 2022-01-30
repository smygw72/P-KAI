import glob

import inference
from config.config import get_config

split_id = 0

def main(log_dir=None):

    cfg = get_config(test_mode=True)

    train_id_path = f'./annotation/{cfg.data.target}/{split_id}/train_id.csv'
    test_id_path = f'./annotation/{cfg.data.target}/{split_id}/test_id.csv'

    train_scores = {}
    for row in open(train_id_path):
        sound_id = row.rstrip()
        print(sound_id)
        sound_path = f'../dataset/{sound_id}.mp3'
        train_scores[sound_id] = inference.main(sound_path, log_dir, 'train')

    test_scores = {}
    for row in open(test_id_path):
        sound_id = row.rstrip()
        print(sound_id)
        sound_path = f'../dataset/{sound_id}.mp3'
        test_scores[sound_id] = inference.main(sound_path, log_dir, 'test')

    train_scores_sorted = sorted(train_scores.items(), key=lambda x: x[1], reverse=True)
    test_scores_sorted = sorted(test_scores.items(), key=lambda x: x[1], reverse=True)

    if log_dir is None:
        save_dir = glob.glob(f'./model/**/split_id={split_id}/', recursive=True)[0]
    else:
        save_dir = log_dir

    with open(f'{save_dir}/train_score.txt', 'w') as file:
        for train_score in train_scores_sorted:
            file.write(f'{train_score[0]} {train_score[1]}\n')
    with open(f'{save_dir}/test_score.txt', 'w') as file:
        for test_score in test_scores_sorted:
            file.write(f'{test_score[0]} {test_score[1]}\n')


if __name__ == "__main__":
    main()
