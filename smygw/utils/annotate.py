import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

annotation_file = './smygw/annotation/all_pair.csv'
# annotation_file = './smygw/test.csv'


def main():
    edges = []  # list of (start, end, weight)
    unannotated_nodes = []

    with open(annotation_file, mode='r') as f:
        reader = csv.reader(f)

        for row in reader:
            if row == ('id1', 'id2', 'label'):
                continue
            if row[2] == 'X':
                unannotated_nodes.append((row[0], row[1]))
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

    if len(unannotated_nodes) == 0:
        print('There are no pairs')
        sys.exit(0)

    df = pd.read_csv(annotation_file)
    for i, node in enumerate(unannotated_nodes):
        dijkstra(i, df, graph, node[0], node[1])
        dijkstra(i, df, graph, node[1], node[0])
    df.to_csv(annotation_file)


def dijkstra(i, df, graph, id1, id2):
    if not graph.has_node(id1):
        return
    if not graph.has_node(id2):
        return
    if nx.has_path(graph, id1, id2):
        length = nx.shortest_path_length(graph, id1, id2, weight='weight')
        answer = question(id1, id2, length)
        df.iloc[i, 2] = answer


def question(id1, id2,length):
    print(f'{id1}, {id2} (length: {length})')
    answer = input() + '  # answered'
    return answer


if __name__ == "__main__":
    main()