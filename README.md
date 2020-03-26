
# TextRank
A simple implementation of the [TextRank algorithm](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf),
tested on research paper abstracts with human annotated key phrases.

## Algorithms
### PageRank
PageRank is an algorithm created by Lawrence Page and Sergey Brin to rank web pages according to their importance,
calculated using the "link to" and "linked by" relationships induced by the hyperlink system. It's implementation
is based on a Markov model, where the limiting distribution of this latter corresponds to so-called scores.

### TextRank
In 2004, the CS department of the University of North Texas came up with a new algorithm, TextRank, based on PageRank.
The idea is the same, but the nodes instead of representing web pages now represent text entities (whatever they might be),
and edges relationships between these entities. The idea behind TextRank was to be able to extract the most important
keywords or key-phrases from text excerpts.

### TF-IDF
Term Frequency - Inverse Document Frequency is a statistical method for computing the most important and relevant
terms in a set of documents. It is a good baseline to compare to other methods, in our case, TextRank.

We enhanced the base algorithm by adding key-phrases concatenation whenever they are adjacent in the text and syntactic
filtering use POS tagging.

## Implementation (Pseudo-code)
### Pre-processing
This step is common to both algorithms. It filters out types of word (verb, determiner, etc) that the user doesn't deem 
relevant or useful and creates all possible n-grams, were n goes from 1 to the number set by the user.
```python
# Inputs: 
# docs: Set of textual documents
# n: n-grams parameter
#
# Output: All possible n-grams up to n from the filtered documents.
tagged_docs := pos_tagging(docs)
filtered_docs := filter_by_tag(docs, syntactic_filters)
return construct_n_grams(filtered_docs, n=i) for i in 1..n  # e.g. for n=3, (I, am, tall) -> (I, am, tall, I am, am tall, I am tall)
```
### TextRank
```python
# Inputs:
# document_ids: Identifiers of all documents
# grams: The grams for each document computed in the pre-processing step
# window_size: Size of the windows of words to calculate co-occurrences
# tol: Convergence criterion.
# k: Maximum number of grams to use for the final key phrases.
#
# Output: The best k (i.e. most relevant) grams for each document
gram_scores = {}
for doc_id in document_ids:
    document_grams = grams[doc_id]
    g = Graph()
    g.nodes = document_grams
    g.edges = (g1, g2), for all g1,g2 in document_grams such that they lie in the same window of size window_size
    node_scores[0] = (1/#nodes)_i for i in 1..#nodes
    t = 1
    do:
        nodes_scores[t] = text_rank_iteration(nodes_scores[t-1])  # As described in the original TextRank paper
    while l1_norm(nodes_scores[t] - nodes_scores[t-1]) > tol
    best_scores = sort_descending(nodes_scores[:k])
    gram_scores[doc_id] = { node such that score(node) is in best_scores }
return gram_scores
```

### TF-IDF
````python
# Inputs:
# document_ids: Identifiers of all documents
# k: Maximum number of grams to use for the final key phrases.
#
# Output: The best k (i.e. most relevant) grams for each document
term_frequencies := nb of occurrences of each word in each document
term_appearances := nb of documents each word appears in
top_k_terms_per_doc = {}
for doc_id in document_ids:
    terms = grams[doc_id]
    scores = term_frequencies[doc_id][term] * log(#documents / term_appearances[term]) for each term in terms
    top_scores = sort_descending(scores)[:k]
    top_k_terms_per_doc[doc_id] = { term,  where score(term) is in top_scores }
return top_k_terms_per_doc
````
### Post-processing
This step is common to both algorithms. It uses the collected terms to create larger key phrases if the terms were adjacent
in the original document.
```python 
# Inputs:
# document_ids: Identifiers of all documents
# terms: Top k terms/grams per document
#
# Output: Most relevant key phrases for each document.
key_phrases
for doc_id in document_ids:
    stack = terms[doc_id]
    key_phrases[doc_id] = List()
    while not stack.is_empty():
        term = stack.pop_first_element()
        old_stack_size = size(stack)
        for other_term in stack:
            if term + " " + other_term is in document:
                stack.push(concat(term, " ", other_term))
            if other_term + " " + term is in document:
                stack.push(concat(other_term, " ", term))
        if size(stack) == old_stack_size:
            key_phrases[doc_id].add(term) 
return key_phrases 
```
## Set-up
The following parameters can be tuned:
- n: maximum length of the grams (1-grams, ..., n-grams) we will use as nodes of the graph.
- k: number of key-phrases to predict per document.
- Syntactic filters: which types of word to consider for key-phrases candidates

We used 2 different metrics to evaluate the models:
- F1-macro score
- Mean Reciprocal Score*

*Since the final scores of key-phrases or not well defined in the original paper, the predictions' order was randomly chosen.

## Results

The same results were achieved using different configurations, so we will only list the most computationally efficient versions of them:


| n | k | Synt. filters | F1-macro | MRR |
| :------: | :-----: | :------: | :-----: | :------: |
| 1 | 3 | Noun/Adj | 0.065 | 0.163 |
| 1 | 1 | Noun/Verb | 0.063 | 0157 |


Tf-idf results for comparison:

| k | F1-macro | MRR |
| :-----: | :-----: | :-----: |
| 1 | 0.031 | 0.109  |
| 3 | 0.083 | 0.195 |
| 5 | 0.095 | 0.209 |
