
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

![preprocessing](https://github.com/arthurdeschamps/TextRank/blob/master/images/preprocessing.png)
### TextRank
![textrank](https://github.com/arthurdeschamps/TextRank/blob/master/images/textrank.png)


### TF-IDF
![tfidf](https://github.com/arthurdeschamps/TextRank/blob/master/images/tfidf.png)
### Post-processing
This step is common to both algorithms. It uses the collected terms to create larger key phrases if the terms were adjacent
in the original document.

![postprocessing](https://github.com/arthurdeschamps/TextRank/blob/master/images/postprocessing.png)
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
