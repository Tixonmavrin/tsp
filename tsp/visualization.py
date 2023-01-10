import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class GraphVisualizer:
    def __init__(self, edge_weight = 2, edge_color = "black", node_size = 20):
        self.Gr = nx.Graph()

        self.edge_weight = edge_weight
        self.edge_color = edge_color
        self.node_size = node_size

    def add_nodes(self, coords):
        for i in range(coords.shape[0]):
            self.Gr.add_node(i)
        return dict(zip(range(coords.shape[0]), coords))

    def add_edges(self, cycle):
        for i in range(len(cycle) - 1):
            self.Gr.add_edge(cycle[i], cycle[i + 1], color=self.edge_color, weight=self.edge_weight)
        self.Gr.add_edge(cycle[len(cycle) - 1], cycle[0], color=self.edge_color, weight=self.edge_weight)

    def draw_graph(self, posGr, ax):
        edges = self.Gr.edges()
        colors = [self.Gr[u][v]["color"] for u, v in edges]
        weights = [self.Gr[u][v]["weight"] for u, v in edges]
        nx.draw(self.Gr, pos=posGr, edge_color=colors, width=weights, ax=ax, node_size=self.node_size)


def visualize_cycles(
    coords: np.ndarray, base_cycle: np.array, best_cycle: np.array
) -> None:
    base_graph_visualizer = GraphVisualizer()
    pos_base = base_graph_visualizer.add_nodes(coords)
    base_graph_visualizer.add_edges(base_cycle)

    best_graph_visualizer = GraphVisualizer()
    pos_best = best_graph_visualizer.add_nodes(coords)
    best_graph_visualizer.add_edges(best_cycle)

    figure, ax = plt.subplots(2)

    base_graph_visualizer.draw_graph(pos_base, ax[0])
    best_graph_visualizer.draw_graph(pos_best, ax[1])

    plt.show()

def visualize_losses(losses):
    plt.plot(losses)
    plt.show()