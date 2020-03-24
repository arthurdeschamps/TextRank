from algorithm.graph_builder import build_graph_n_grams
from algorithm.text_rank import TextRank

text = "Compatibility of systems of linear constraints over the set of natural numbers. \
Criteria of compatibility of a system of linear Diophantine equations, strict \
inequations, and nonstrict inequations are considered. Upper bounds for \
components of a minimal set of solutions and algorithms of construction of \
minimal generating sets of solutions for all types of systems are given. \
These criteria and the corresponding algorithms for constructing a minimal \
supporting set of solutions can be used in solving all the considered types \
systems and systems of mixed types"

g = build_graph_n_grams(text, n=1)
tr = TextRank(g)
tr.run()
print(tr.best_nodes(limit=5))
