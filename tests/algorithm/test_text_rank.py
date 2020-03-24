import networkx as nx
import pytest

from algorithm.text_rank import TextRank


def test_unweighted_directed_page_rank():
    graph = nx.DiGraph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_weighted_edges_from([(0, 1, 0.5), (1, 0, 0.5), (0, 0, 0.5), (1, 1, 0.5)])
    pr = TextRank(graph, damping_factor=1.0)
    scores = pr.run()
    assert (0.5 == pytest.approx(scores[0], 1e-4)) and (0.5 == pytest.approx(scores[1], 1e-4))


def test_unweighted_undirected_page_rank():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4])
    for i in range(4):
        for j in range(4):
            if i != j:
                graph.add_edge(i + 1, j + 1, weight=1/3.0)
    scores = TextRank(graph, damping_factor=1.0).run()
    for score in scores:
        assert score == pytest.approx(0.25, 1e-4)


def test_weighted_directed_page_rank():
    graph = nx.DiGraph()
    graph.add_nodes_from([1, 2])
    a = 0.223
    b = 0.69
    graph.add_weighted_edges_from([
        (1, 1, 1-a),
        (2, 2, 1-b),
        (2, 1, b),
        (1, 2, a)
    ])
    scores = TextRank(graph, damping_factor=1.0).run()
    assert scores[0] == pytest.approx(b/(a+b), 1e-1)
    assert scores[1] == pytest.approx(a/(a+b), 1e-1)
