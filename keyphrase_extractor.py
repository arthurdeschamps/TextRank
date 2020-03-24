from functools import reduce

from algorithm.graph_builder import build_graph_n_grams
from algorithm.text_rank import TextRank


class KeyphraseExtractor:

    def __init__(self, text):
        super(KeyphraseExtractor, self).__init__()
        self.text = text.lower()
        self.graph = build_graph_n_grams(text, n=1)

    def extract_keyphrases(self):
        nb_nodes = self.graph.number_of_nodes()
        tr = TextRank(self.graph)
        tr.run()
        # Only keeping 1/3 of the nodes (see original TextRank paper)
        best_nodes = list(phrase for (phrase, score) in tr.best_nodes(limit=int(nb_nodes/3)))
        keyphrases = self._collapse_adjacent_keyphrases(best_nodes)
        return keyphrases

    def _collapse_adjacent_keyphrases(self, keyphrases):
        final_keywords = []

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
                if concatenated_v1 in self.text:
                    concatenated_keyphrases.append(concatenated_v1)
                    already_collapsed.add(keyphrases_stack[i])
                elif concatenated_v2 in self.text:
                    concatenated_keyphrases.append(concatenated_v2)
                    already_collapsed.add(keyphrases_stack[i])

            if len(concatenated_keyphrases) > 0:
                keyphrases_stack += concatenated_keyphrases
            elif keyphrase not in already_collapsed:
                final_keywords.append(keyphrase)
        return final_keywords



