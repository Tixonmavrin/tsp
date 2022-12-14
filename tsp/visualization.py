import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def add_nodes(Gr, coords):
    pos = dict(zip(range(coords.shape[0]), coords))
    for i in range(coords.shape[0]):
        Gr.add_node(i)
    return pos


def add_edges(Gr, cycle):
    for i in range(len(cycle) - 1):
        Gr.add_edge(cycle[i], cycle[i + 1], color="black", weight=2)
    Gr.add_edge(cycle[len(cycle) - 1], cycle[0], color="black", weight=2)


def draw_graph(Gr, posGr, ax):
    edges = Gr.edges()
    colors = [Gr[u][v]["color"] for u, v in edges]
    weights = [Gr[u][v]["weight"] for u, v in edges]
    nx.draw(Gr, pos=posGr, edge_color=colors, width=weights, ax=ax, node_size=20)


def visualize_cycles(
    coords: np.ndarray, greedy_cycle: np.array, best_cycle: np.array
) -> None:
    G = nx.Graph()
    posG = add_nodes(G, coords)
    add_edges(G, greedy_cycle)

    E = nx.Graph()
    posE = add_nodes(E, coords)
    add_edges(E, best_cycle)

    figure, ax = plt.subplots(2)

    draw_graph(G, posG, ax[0])
    draw_graph(E, posE, ax[1])

    plt.show()
