from typing import List, Tuple
import numpy as np
import networkx as nx


class TextRank:
    """
    Implementation of the TextRank algorithm.
    """

    def __init__(self, graph: nx.DiGraph or nx.Graph, damping_factor: float = 0.85):
        super(TextRank, self).__init__()
        self.graph = graph
        self.damping_factor = damping_factor
        self.directed = graph.is_directed()
        self.scores: List[float] = []
        assert nx.is_weighted(graph)

    def run(self, delta_tol=1e-6) -> List[Tuple[any, float]]:
        """
        Runs the TextRank algorithm on the current graph.
        :param delta_tol: Convergence criterion (||x_t - x_(t-1)||_1 < delta)
        :return: A list containing all the (node, node_score) pairs.
        """
        self._init_nodes_scores()
        return self._run(0, delta_tol)

    def best_nodes(self, limit=10):
        """
        :param limit: Maximum number of  pairs to retrieve.
        :return: The best (node, node_score) pairs.
        """
        assert len(self.scores) > 0
        return sorted(self.scores, key=lambda t: t[1], reverse=True)[:limit]

    def _run(self, it, tol):
        d_comp = 1.0 - self.damping_factor
        for node in self.graph.nodes:
            predecessors = self.graph.predecessors(node) if self.directed else self.graph.neighbors(node)
            new_score = 0.0
            for neighbor in predecessors:
                successors = self.graph.successors(neighbor) if self.directed else self.graph.neighbors(neighbor)
                edge_weight = self.graph.edges[neighbor, node]['weight']
                sum_weights = sum(self.graph.edges[neighbor, adj_node]['weight'] for adj_node in successors)
                new_score += edge_weight * self.graph.nodes[neighbor]['previous_score'] / sum_weights
            self.graph.nodes[node]['score'] = d_comp + self.damping_factor * new_score
        delta = self._compute_delta()
        if delta < tol:
            self.nb_iterations = it + 1
            self.scores = list((node, data['score']) for node, data in self.graph.nodes(data=True))
            return self.scores
        self._update_previous_scores()
        return self._run(it+1, tol)

    def _compute_delta(self):
        delta = 0.0
        for _, node in self.graph.nodes(data=True):
            delta += np.abs(node['previous_score'] - node['score'])
        return delta

    def _update_previous_scores(self):
        for ind, node in self.graph.nodes(data=True):
            node['previous_score'] = node['score']
            # print(f"{ind}: {node['score']}")

    def _init_nodes_scores(self):
        uniform_distr = np.full(shape=(self.graph.number_of_nodes(),), fill_value=1/len(self.graph.nodes))
        i = 0
        for _, node in self.graph.nodes(data=True):
            node['score'] = uniform_distr[i]
            node['previous_score'] = node['score']
            i += 1
