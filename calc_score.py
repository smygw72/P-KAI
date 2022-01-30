import glob

split_id = 0

def read_score(files):
    scores = {}
    standard = []

    with open(files, 'r') as f:
        rows = f.read().splitlines()

    for row in rows:
        cols = row.split(' ')
        sound_id = cols[0]
        relative_score = float(cols[1])
        absolute_score = None
        if len(cols) == 3:
            absolute_score = float(cols[2])
            standard.append((relative_score, absolute_score))
        scores[sound_id] = [relative_score, absolute_score]
    return scores, standard


def predict_absolute_score(rel_score, standard):
    if rel_score > standard[0][0]:
        return standard[0][0]
    elif rel_score < standard[-1][0]:
        return standard[-1][0]
    else:
        for i in range(len(standard) - 1):
            lower_rel_score = standard[i + 1][0]
            upper_rel_score = standard[i][0]
            lower_abs_score = standard[i + 1][1]
            upper_abs_score = standard[i][1]
            if (rel_score > lower_rel_score) and (rel_score < upper_rel_score):
                ratio = (rel_score - lower_rel_score) / (upper_rel_score - lower_rel_score)
                return ratio * (upper_abs_score - lower_abs_score) + lower_abs_score


def main(log_dir=None):

    if log_dir is None:
        save_dir = glob.glob(f'./model/**/split_id={split_id}/', recursive=True)[0]
    else:
        save_dir = log_dir

    train_scores, standard = read_score(f'{save_dir}/train_score.txt')
    test_scores, __ = read_score(f'{save_dir}/test_score.txt')

    for key, value in train_scores.items():
        if value[1] is None:
            value[1] = predict_absolute_score(value[0], standard)

    for key, value in test_scores.items():
        value[1] = predict_absolute_score(value[0], standard)

    with open(f'{save_dir}/train_score_completed.txt', 'w') as file:
        for key, value in train_scores.items():
            file.write(f'{key} {value[0]} {value[1]}\n')
    with open(f'{save_dir}/test_score_completed.txt', 'w') as file:
        for key, value in test_scores.items():
            file.write(f'{key} {value[0]} {value[1]}\n')



if __name__ == "__main__":
    main()
