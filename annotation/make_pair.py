import csv
import pandas as pd

from config import CONFIG


def main(*args, **kwargs):
    pair_list = []
    df = pd.read_csv(CONFIG.common.datalist_path, header=0)
    ids = df["ID"]

    for i, id1 in enumerate(ids):
        for j, id2 in enumerate(ids):
            if i < j:
                pair_list.append({'id1': id1, 'id2': id2})

    print(len(pair_list))

    with open(CONFIG.common.allpair_path, mode='r') as f:
        reader = csv.reader(f)
        old_list = []
        for row in reader:
            old_id1 = row[0]
            old_id2 = row[1]
            old_row = {'id1': old_id1, 'id2': old_id2}
            old_list.append(old_row)

    with open(CONFIG.common.allpair_path, mode='a') as f:
        writer = csv.writer(f)

        if len(old_list) == 0:
            writer.writerow(['id1', 'id2', 'label'])  # header

        for pair in pair_list:
            new_id1 = pair["id1"]
            new_id2 = pair["id2"]
            new_row = {'id1': new_id1, 'id2': new_id2}
            if new_row not in old_list:
                writer.writerow([new_id1, new_id2, 'X'])


if __name__ == "__main__":
    main()
