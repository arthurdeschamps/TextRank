from typing import List
import numpy as np
import networkx as nx


class TextRank:

    def __init__(self, graph: nx.DiGraph, damping_factor: float):
        super(TextRank, self).__init__()
        self.graph = graph
        self.damping_factor = damping_factor
        self.directed = graph.is_directed()
        assert nx.is_weighted(graph)

    def run(self, delta_tol=1e-6) -> List[float]:
        self._init_nodes_scores()
        return self._run(0, delta_tol)

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
            return list(node['score'] for _, node in self.graph.nodes(data=True))
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
        uniform_distr = np.full(shape=(len(self.graph.nodes),), fill_value=1/len(self.graph.nodes))
        i = 0
        for _, node in self.graph.nodes(data=True):
            node['score'] = uniform_distr[i]
            node['previous_score'] = node['score']
            i += 1
