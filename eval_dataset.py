import inference
from config.config import get_config


def main(log_dir=None):

    cfg = get_config(test_mode=True)

    split_id=0
    train_id_path = f'./annotation/{cfg.data.target}/{split_id}/train_id.csv'
    test_id_path = f'./annotation/{cfg.data.target}/{split_id}/test_id.csv'

    scores = {}
    for row in open(train_id_path):
        sound_id = row.rstrip()
        print(sound_id)
        sound_path = f'../dataset/{sound_id}.mp3'
        scores[sound_id] = inference.main(sound_path, log_dir, 'train')

    for row in open(test_id_path):
        sound_id = row.rstrip()
        print(sound_id)
        sound_path = f'../dataset/{sound_id}.mp3'
        scores[sound_id] = inference.main(sound_path, log_dir, 'test')

    score_sorted = sorted(scores.items(), key=lambda x: x[1])
    print(score_sorted)


if __name__ == "__main__":
    main()
