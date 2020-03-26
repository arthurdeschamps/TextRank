from typing import List, Tuple

import nltk
import numpy as np

from utils import apply_syntactic_filters


class TFIDF:

    def __init__(self, documents, syntactic_filters=('NOUN', 'ADJ')):
        super(TFIDF, self).__init__()
        self.documents_with_ids = list((i, documents[i]) for i in range(len(documents)))
        self.syntactic_filters = syntactic_filters
        # Frequencies of term for each document
        self.term_frequencies = {}
        # In how many documents does a specific term appear
        self.term_appearances = {}
        self._compute_term_frequencies()

    def _compute_term_frequencies(self):
        for doc_id, document in self.documents_with_ids:
            self.term_frequencies[doc_id] = {}
            tokens = nltk.word_tokenize(document)
            tokens = apply_syntactic_filters(nltk.pos_tag(tokens), self.syntactic_filters)
            unique, counts = np.unique(tokens, return_counts=True)
            for word, frequency in zip(unique, counts):
                self.term_frequencies[doc_id][word] = frequency
                if not(word in self.term_appearances):
                    self.term_appearances[word] = set()
                self.term_appearances[word].add(doc_id)

    def get_keyphrases(self, top_k=3) -> List[List[Tuple[str, float]]]:
        """
        :param top_k: How many words to use from the text to construct key-phrases.
        :return: Lists of key phrases along with their scores for each document in the document set.
        """
        top_keyphrases_per_document = []
        for doc_id, document in self.documents_with_ids:
            scores = []
            for term in self.term_frequencies[doc_id].keys():
                scores.append((term, self.tf_idf_score(term, doc_id)))
            top_scores = sorted(scores, key=lambda t: t[1], reverse=True)[:top_k]
            top_keyphrases_per_document.append(top_scores)
        return top_keyphrases_per_document

    def tf_idf_score(self, term, doc_id):
        return self.term_frequencies[doc_id][term] * self.inverse_document_frequency(term)

    def inverse_document_frequency(self, term: str):
        return np.log(len(self.documents_with_ids) / len(self.term_appearances[term]))
