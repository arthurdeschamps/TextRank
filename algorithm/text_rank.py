from typing import List
import numpy as np
import networkx as nx


class TextRank:

    def __init__(self, graph: nx.Graph or nx.DiGraph, damping_factor: float):
        super(TextRank, self).__init__()
        self.graph = graph
        self.damping_factor = damping_factor
        self.directed = graph.is_directed()
        self.weighted = nx.is_weighted(graph)

    def run(self, delta_tol=1e-6) -> List[float]:
        self._init_nodes_scores()
        return self._run(0, delta_tol)

    def _run(self, it, tol):
        d_comp = 1.0 - self.damping_factor
        if self.weighted:
            raise NotImplementedError()
        else:
            for node in self.graph.nodes:
                predecessors = self.graph.predecessors(node) if self.directed else self.graph.neighbors(node)
                new_score = 0.0
                for neighbour in predecessors:
                    out_degree = self.graph.out_degree(neighbour) if self.directed else self.graph.degree(neighbour)
                    new_score += self.graph.nodes[neighbour]['previous_score'] / out_degree
                self.graph.nodes[node]['score'] = d_comp + self.damping_factor * new_score
        delta = self._compute_delta()
        if delta < tol:
            self.nb_iterations = it + 1
            return list(node['score'] for _, node in self.graph.nodes(data=True))
        self._update_previous_scores()
        return self._run(it+1, tol)

    def _compute_delta(self):
        delta = 0.0
        for _, node in self.graph.nodes(data=True):
            delta += np.abs(node['previous_score'] - node['score'])
        return delta

    def _update_previous_scores(self):
        for _, node in self.graph.nodes(data=True):
            node['previous_score'] = node['score']

    def _init_nodes_scores(self):
        for _, node in self.graph.nodes(data=True):
            node['score'] = np.random.uniform(low=0.0, high=10.0)
            node['previous_score'] = node['score']
