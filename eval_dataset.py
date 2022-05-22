import glob

import inference
from config.config import get_config

split_id = 0

def main(log_dir=None):

    cfg = get_config(test_mode=True)

    if log_dir is None:
        save_dir = glob.glob(f'./model/**/split_id={split_id}/', recursive=True)[0]
    else:
        save_dir = log_dir

    # train_id_path = f'./annotation/{cfg.data.target}/{split_id}/train_id.csv'
    # get_scores(train_id_path, target='train')
    # test_id_path = f'./annotation/{cfg.data.target}/{split_id}/test_id.csv'
    # get_scores(test_id_path, target='test')
    train_id_path = f'./annotation/psa/datalist.csv'
    get_scores(train_id_path, target='psa_all')


def get_scores(id_path, target):
    scores = {}
    for row in open(id_path):
        sound_id = row.rstrip()
        print(sound_id)
        sound_path = f'../dataset/{sound_id}.mp3'
        scores[sound_id] = inference.main(sound_path, log_dir)
    scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    with open(f'{save_dir}/{target}.txt', 'w') as file:
        for score in scores_sorted:
            file.write(f'{score[0]} {score[1]}\n')
    return scores


if __name__ == "__main__":
    main()
