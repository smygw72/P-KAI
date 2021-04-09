import csv
import pandas as pd


def main():
    pair_list = []
    df = pd.read_csv('./smygw/youtube.csv', header=0)
    ids = df["ID"]

    for i, id1 in enumerate(ids):
        for j, id2 in enumerate(ids):
            if i < j:
                pair_list.append({'id1': id1, 'id2': id2})
    print(len(pair_list))

    with open('./smygw/annotation/all_pair.csv', mode='a') as f:
        reader = csv.reader(f)
        old_list = []
        for row in reader:
            old_id1 = row[0]
            old_id2 = row[1]
            old_row = {'id1': old_id1, 'id2': old_id2}
            old_list.append(old_row)

        writer = csv.writer(f)

        if len(old_list) > 0:
            writer.writerow(['id1', 'id2', 'label'])  # header

        for pair in pair_list:
            new_id1 = pair["id1"]
            new_id2 = pair["id2"]
            new_row = {'id1': new_id1, 'id2': new_id2}
            if not new_row in old_list:
                writer.writerow([new_id1, new_id2, 'X'])


if __name__ == "__main__":
    main()