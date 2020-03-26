import nltk
from nltk.tag import map_tag, pos_tag
import networkx as nx
from utils import visualize_graph, apply_syntactic_filters


def build_graph_n_grams(from_text: str, syntactic_filters=('NOUN', 'ADJ'), n=2, window_size=2):
    """

    :param window_size:
    :param from_text:
    :param syntactic_filters: One or many of ADJ, ADP, ADV, CONJ, DET, NOUN, NUM, PRT, PRON,
    VERB, ., X
    :param n:
    :return:
    """
    tokens = nltk.word_tokenize(from_text)
    pos_tags = pos_tag(tokens)
    filtered_tags = apply_syntactic_filters(pos_tags, syntactic_filters)

    graph = nx.Graph()
    # the grams will become nodes of the graph
    n_grams = (list(nltk.ngrams(filtered_tags, i)) for i in range(1, n+1))
    for grams in n_grams:
        for i in range(len(grams)):
            graph.add_node(grams[i])
            for j in range(i-window_size+1, i+window_size):
                if (i != j) and (j >= 0) and (j < len(grams)):
                    # Within the same window, we add an edge
                    graph.add_edge(grams[i], grams[j], weight=1.0)
    return graph
