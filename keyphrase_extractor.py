from functools import reduce

from algorithm.graph_builder import build_graph_n_grams
from algorithm.text_rank import TextRank
from utils import collapse_adjacent_keyphrases


class KeyphraseExtractor:

    def __init__(self, text, n=1, syntactic_filters=('NOUN', 'ADJ')):
        super(KeyphraseExtractor, self).__init__()
        self.text = text.lower()
        self.graph = build_graph_n_grams(text, n=n, syntactic_filters=syntactic_filters)

    def extract_keyphrases(self, nb_keyphrases=None):
        nb_nodes = self.graph.number_of_nodes()
        if not nb_keyphrases:
            nb_keyphrases = int(nb_nodes/3)
        tr = TextRank(self.graph)
        tr.run()
        # Only keeping 1/3 of the nodes (see original TextRank paper)
        best_nodes = list(phrase for (phrase, score) in tr.best_nodes(limit=nb_keyphrases))
        keyphrases = collapse_adjacent_keyphrases(self.text, best_nodes)
        return keyphrases





