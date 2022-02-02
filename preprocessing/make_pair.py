import csv
import argparse
import pandas as pd


def main(*args, **kwargs):

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['all', 'elise', 'psa'])
    args = parser.parse_args()
    target = args.target

    pair_list = []

    datalist_path = f'./annotation/{target}/datalist.txt'
    with open(datalist_path, 'r') as f:
        data_list = f.readlines()
    for i in range(len(data_list)):
        data_list[i] = data_list[i].replace('\n', '')

    for i, id1 in enumerate(data_list):
        for j, id2 in enumerate(data_list):
            if i < j:
                pair_list.append({'id1': id1, 'id2': id2})

    print(len(pair_list))

    with open(f'./annotation/{target}/all_pair.csv', mode='r') as f:
        reader = csv.reader(f)
        old_list = []
        for row in reader:
            old_id1 = row[0]
            old_id2 = row[1]
            old_row = {'id1': old_id1, 'id2': old_id2}
            old_list.append(old_row)

    with open(f'./annotation/{target}/all_pair.csv', mode='w') as f:
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
