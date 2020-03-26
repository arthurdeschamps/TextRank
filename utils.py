from functools import reduce
from typing import List, Tuple

import networkx as nx
import pandas as pd
import os

from nltk import map_tag


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


def mean_reciprocal_rank(predicted_keyphrases, gold_keyphrases):
    if len(predicted_keyphrases) < 1:
        return 0.0
    for i in range(len(predicted_keyphrases)):
        if predicted_keyphrases[i] in gold_keyphrases:
            return 1.0 / (i + 1)
    return 0.0


def collapse_adjacent_keyphrases(text, keyphrases: List[Tuple[str]]):
    final_keywords = []

    if len(keyphrases) == 1:
        return keyphrases

    def mk_str(phrase) -> str:
        if len(phrase) == 1:
            return phrase[0]
        return reduce(lambda k1, k2: k1 + " " + k2, phrase)

    keyphrases_stack = list(mk_str(keyphrase) for keyphrase in keyphrases)
    already_collapsed = set()
    while len(keyphrases_stack) > 0:
        concatenated_keyphrases = []
        keyphrase = keyphrases_stack.pop(0)

        for i in range(len(keyphrases_stack)):
            concatenated_v1 = mk_str((keyphrase, keyphrases_stack[i]))
            concatenated_v2 = mk_str((keyphrases_stack[i], keyphrase))
            if concatenated_v1 in text:
                concatenated_keyphrases.append(concatenated_v1)
                already_collapsed.add(keyphrases_stack[i])
            elif concatenated_v2 in text:
                concatenated_keyphrases.append(concatenated_v2)
                already_collapsed.add(keyphrases_stack[i])

        if len(concatenated_keyphrases) > 0:
            keyphrases_stack += concatenated_keyphrases
        elif keyphrase not in already_collapsed:
            final_keywords.append(keyphrase)
    return final_keywords


def apply_syntactic_filters(pos_tagged_tokens, syntactic_filters):
    tags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in pos_tagged_tokens]
    return [word.lower() for (word, tag) in tags if tag in syntactic_filters]
