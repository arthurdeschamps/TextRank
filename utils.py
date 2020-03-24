import networkx as nx
import pandas as pd
import os


def visualize_graph(graph: nx.Graph or nx.DiGraph):
    import matplotlib.pyplot as plt
    plt.plot()
    nx.draw_shell(graph, with_labels=True)
    plt.show()


def get_dataset():
    ds = []
    base_dir = f"{os.path.dirname(os.path.abspath(__file__))}/data/kptest"
    query_indices = pd.read_csv(f"{base_dir}/queries.all.list", header=None, names=['index'])['index']

    for ind in query_indices:
        try:
            abstract = open(f"{base_dir}/contentsubset/{ind}", "r").read()
            gold_keywords = pd.read_csv(f"{base_dir}/gold/{ind}", header=None, names=['keyphrases'])['keyphrases'].tolist()
            ds.append((abstract, gold_keywords))
        except FileNotFoundError:
            pass
    return ds


def f1_macro(predicted_keyphrases, gold_keyphrases):
    if len(predicted_keyphrases) < 1:
        return 0.0
    overlap = len(list(set(predicted_keyphrases) & set(gold_keyphrases)))
    if overlap == 0:
        return 0.0
    precision = overlap / float(len(predicted_keyphrases))
    recall = overlap / float(len(gold_keyphrases))
    return 2 * (precision * recall) / (precision + recall)
