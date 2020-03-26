from keyphrase_extractor import KeyphraseExtractor
from utils import get_dataset, f1_macro, mean_reciprocal_rank


def run_exp(ds, n, nb_keyphrases, syntactic_filters):
    print(f"Parameters: n -> {n} - Keyphrases/prediction -> {nb_keyphrases} - Syntactic filters -> {syntactic_filters}")
    f1_avg = 0.0
    mrr_avg = 0.0
    for abstract, gold_keyphrases in ds:
        keyphrases = KeyphraseExtractor(abstract, n=2, syntactic_filters=syntactic_filters)\
            .extract_keyphrases(nb_keyphrases=5)
        f1_avg += f1_macro(keyphrases, gold_keyphrases) / len(ds)
        mrr_avg += mean_reciprocal_rank(keyphrases, gold_keyphrases) / len(ds)
    print(f"F1 macro: {f1_avg}")
    print(f"Mean Reciprocal Rank: {mrr_avg}")


def grid_search():
    ds = get_dataset()
    grid = {
        "n": [1, 2, 3],
        "nb_keyphrases": [1, 3, 5, None],
        "syntactic_filters": [("NOUN", "ADJ"), ("NOUN", "VERB"), ("NOUN", "ADJ", "ADV")]
    }
    for n in grid["n"]:
        for nb_keyphrases in grid["nb_keyphrases"]:
            for syntactic_filters in grid["syntactic_filters"]:
                run_exp(ds, n, nb_keyphrases, syntactic_filters)


run_exp(get_dataset(), 1, None, ('NOUN', 'ADJ', 'ADV'))

