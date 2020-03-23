import networkx as nx
import pytest
from algorithm.text_rank import TextRank


def test_unweighted_directed_page_rank():
    graph = nx.DiGraph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edges_from([(0, 1), (1, 0), (0, 0), (1, 1)])
    pr = TextRank(graph, damping_factor=0.85)
    scores = pr.run()
    assert (1.0 == pytest.approx(scores[0], 1e-4)) and (1.0 == pytest.approx(scores[1], 1e-4))


def test_unweighted_undirected_page_rank():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4])
    for i in range(4):
        for j in range(4):
            if i != j:
                graph.add_edge(i + 1, j + 1)
    scores = TextRank(graph, damping_factor=0.85).run()
    for score in scores:
        assert score == pytest.approx(1.0, 1e-4)


def test_weighted_undirected_page_rank():
    pass


def test_weighted_directed_page_rank():
    graph = nx.DiGraph()
    graph.add_nodes_from([1, 2, 3])
    graph.add_weighted_edges_from([
        (1, 1, 0.4),
        (1, 2, 0.3),
        (1, 3, 0.3),
        (2, 1, 0.5),
        (2, 2, 0.3),
        (2, 3, 0.2),
        (3, 1, 0.1),
        (3, 2, 0.4),
        (3, 3, 0.5)
    ])
    scores = TextRank(graph, damping_factor=0.85).run()
    for score in scores:
        assert score == pytest.approx(1/3.0, 1e-4)
