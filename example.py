from tsp import TSP, NegativeCycleException, UnreachableVertexException
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class GraphPlanarGenerator:
    def __init__(self, n = 10, loc = 0.0, scale = 1.0):
        self.n = n
        self.loc = loc
        self.scale = scale

    def generate(self):
        coords = np.random.normal(loc=self.loc, scale=self.scale, size=(self.n, 2))
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        return coords, dist_matrix

if __name__ == "__main__":
    generator = GraphPlanarGenerator()
    
    coords, dist_matrix = generator.generate()
    tsp = TSP(dist_matrix)
    try:
        greedy_distance, greedy_cycle, shortest_distance, shortest_cycle = tsp.solve()

        print("Кратчайшее расстояние из жадного алгоритма:", greedy_distance)
        print("Длина цикла из жадного алгоритма:", len(greedy_cycle))
        print("Цикл из жадного алгоритма:", greedy_cycle)
        print("Итоговое кратчайшее расстояние:", shortest_distance)
        print("Итоговая длина цикла:", len(shortest_cycle))
        print("Итоговый цикл:", shortest_cycle)


        G=nx.Graph()
        pos = dict(zip(range(coords.shape[0]), coords))
        for i in range(coords.shape[0]):
            G.add_node(i)

        for i in range(len(greedy_cycle) - 1):
            G.add_edge(greedy_cycle[i], greedy_cycle[i + 1], color="black", weight=3)
        G.add_edge(greedy_cycle[len(greedy_cycle) - 1], greedy_cycle[0], color="black", weight=3)

        E=nx.Graph()
        pos = dict(zip(range(coords.shape[0]), coords))
        for i in range(coords.shape[0]):
            E.add_node(i)

        for i in range(len(shortest_cycle) - 1):
            E.add_edge(shortest_cycle[i], shortest_cycle[i + 1], color="black", weight=3)
        E.add_edge(shortest_cycle[len(shortest_cycle) - 1], shortest_cycle[0], color="black", weight=3)

        edges = G.edges()
        colors = [G[u][v]['color'] for u,v in edges]
        weights = [G[u][v]['weight'] for u,v in edges]

        figure, ax = plt.subplots(2)

        nx.draw(G, pos=pos, with_labels=True, edge_color=colors, width=weights, ax=ax[0])

        edges = E.edges()
        colors = [E[u][v]['color'] for u,v in edges]
        weights = [E[u][v]['weight'] for u,v in edges]
        nx.draw(E, pos=pos, with_labels=True, edge_color=colors, width=weights, ax=ax[1])

        plt.show()

    except NegativeCycleException as ex:
        print(ex)
    except UnreachableVertexException as ex:
        print(ex)
