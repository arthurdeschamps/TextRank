from algorithm.tf_idf import TFIDF
from utils import get_dataset, f1_macro, mean_reciprocal_rank, collapse_adjacent_keyphrases


def run_exp(gold_keyphrases, top_k, tf_idf: TFIDF):
    print(f"Top {top_k} key phrases")
    predicted_keyphrases = tf_idf.get_keyphrases(top_k)
    f1_avg = 0.0
    mrr_avg = 0.0

    for i in range(len(predicted_keyphrases)):
        keyphrases_with_scores = predicted_keyphrases[i]
        if len(keyphrases_with_scores) > 1:
            keyphrases = list((keyphrase,) for keyphrase, _ in keyphrases_with_scores)
            keyphrases = collapse_adjacent_keyphrases(tf_idf.documents_with_ids[i][1], keyphrases)
        else:
            keyphrases = keyphrases_with_scores[0]
        f1_avg += f1_macro(keyphrases, gold_keyphrases[i]) / len(gold_keyphrases)
        mrr_avg += mean_reciprocal_rank(keyphrases, gold_keyphrases[i]) / len(gold_keyphrases)
    print(f"F1 macro: {f1_avg}")
    print(f"Mean Reciprocal Rank: {mrr_avg}")


def line_search():
    ds = get_dataset()
    documents = list(doc for doc, _ in ds)
    gold_keyphrases = list(keyphrases for _, keyphrases in ds)
    tf_idf = TFIDF(documents)
    for k in (1, 3, 5):
        run_exp(gold_keyphrases, k, tf_idf)


line_search()
