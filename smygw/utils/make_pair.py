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
        writer = csv.writer(f)
        writer.writerow(['id1', 'id2', 'label'])  # header
        for pair in pair_list:
            writer.writerow([pair["id1"], pair["id2"], 'X'])


if __name__ == "__main__":
    main()
