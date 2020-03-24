import networkx as nx


def visualize_graph(graph: nx.Graph or nx.DiGraph):
    import matplotlib.pyplot as plt
    plt.plot()
    #nx.draw(graph, with_labels=True)
    nx.draw_shell(graph, with_labels=True)
    plt.show()
