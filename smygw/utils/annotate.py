import argparse
import sys
import csv
import pandas as pd
import networkx as nx

import _paths
from config import CONFIG


miyagawa_range = (1, 145)
kanaishi_range = (146, 290)
kuroki_range = (291, 435)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--miyagawa', action="store_true")
    parser.add_argument('--kanaishi', action="store_true")
    parser.add_argument('--kuroki', action="store_true")
    return parser.parse_args()


def main(*args, **kwargs):
    args = args_parser()
    edges = []  # list of (start, end, weight)
    unannotated_idxs = []

    with open(CONFIG.common.datalist_path, mode='r') as f:
        reader = csv.reader(f)

        for i, row in enumerate(reader):
            if row == ('id1', 'id2', 'label'):
                continue
            if row[2] == 'X':
                if (args.miyagawa is True) and ((i < miyagawa_range[0]) or (i > miyagawa_range[1])):
                    continue
                elif (args.kanaishi is True) and ((i < kanaishi_range[0]) or (i > kanaishi_range[1])):
                    continue
                elif (args.kuroki is True) and ((i < kuroki_range[0]) or (i > kuroki_range[1])):
                    continue
                else:
                    unannotated_idxs.append(i - 1)
            else:
                label = row[2]
                if label == '1':
                    edges.append((row[0], row[1], 1))
                elif label == '-1':
                    edges.append((row[1], row[0], 1))
                elif label == '0':
                    edges.append((row[0], row[1], 0))
                    edges.append((row[1], row[0], 0))

    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges)
    # nx.draw_networkx(graph, with_labels=True, node_color='red', alpha=0.5)
    # plt.axis("off")
    # plt.show()

    if len(unannotated_idxs) == 0:
        print('There are no pairs')
        sys.exit(0)

    df = pd.read_csv(CONFIG.common.allpair_path)
    print("Input 1/0/-1\n")

    for idx in unannotated_idxs:
        node1 = df.iloc[idx, 0]
        node2 = df.iloc[idx, 1]

        answer = question(graph, node1, node2)
        if answer is not None:
            df.iloc[idx, 2] = str(answer)
            df.to_csv(CONFIG.common.allpair_path, index=False)
            continue
        answer = question(graph, node2, node1)
        if answer is not None:
            df.iloc[idx, 2] = str(-1 * answer)
            df.to_csv(CONFIG.common.allpair_path, index=False)
            continue


def question(graph, id1, id2):
    if not graph.has_node(id1):
        return None
    if not graph.has_node(id2):
        return None
    if nx.has_path(graph, id1, id2):
        length = nx.shortest_path_length(graph, id1, id2, weight='weight')
        print(f'{id1}, {id2} (distance: {length})')
        return int(input())
    return None


if __name__ == "__main__":
    main()
