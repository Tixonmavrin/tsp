from tsp import TSP, NegativeCycleException, UnreachableVertexException
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class GraphPlanarGenerator:
    def __init__(self, n = 20, loc = 0.0, scale = 1.0):
        self.n = n
        self.loc = loc
        self.scale = scale

    def generate(self):
        coords = np.random.normal(loc=self.loc, scale=self.scale, size=(self.n, 2))
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        return coords, dist_matrix

def visualize_cycles(coords: np.ndarray, greedy_cycle: np.array, best_cycle: np.array) -> None:
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

    for i in range(len(best_cycle) - 1):
        E.add_edge(best_cycle[i], best_cycle[i + 1], color="black", weight=3)
    E.add_edge(best_cycle[len(best_cycle) - 1], best_cycle[0], color="black", weight=3)

    figure, ax = plt.subplots(2)

    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    weights = [G[u][v]['weight'] for u,v in edges]
    nx.draw(G, pos=pos, edge_color=colors, width=weights, ax=ax[0])

    edges = E.edges()
    colors = [E[u][v]['color'] for u,v in edges]
    weights = [E[u][v]['weight'] for u,v in edges]
    nx.draw(E, pos=pos, edge_color=colors, width=weights, ax=ax[1])

    plt.show()

class Operation:
    def __init__(self, n_vertex, first_index_fn=None, delta_index_fn=None, p_generator=None):
        if first_index_fn is None:
            first_index_fn = lambda: np.random.randint(0, n_vertex)
        self.first_index_fn = first_index_fn

        if delta_index_fn is None:
            delta_index_fn = lambda: np.random.randint(0, n_vertex)
        self.delta_index_fn = delta_index_fn

        self.p_generator = lambda: np.random.binomial(n=1, p=0.5)

    def get_index(self, i, length):
        return i % length

    def reverse(self, cycle, index, other_index):
        while index != other_index and \
              index != self.get_index(other_index - 1, len(cycle)):
            cycle[index], cycle[other_index] = cycle[other_index], cycle[index]
            index = self.get_index(index + 1, len(cycle))
            other_index = self.get_index(other_index - 1, len(cycle))

        if index == self.get_index(other_index - 1, len(cycle)):
            cycle[index], cycle[other_index] = cycle[other_index], cycle[index]

    def __call__(self, cycle, distances, distance, dist_matrix):
        index = self.get_index(self.first_index_fn(), len(cycle))
        delta = self.delta_index_fn()
        other_index = self.get_index(index + delta, len(cycle))
        distance_diff = 0

        if self.p_generator():
            # Swap
            distance_diff -= distances[self.get_index(index - 1, len(cycle))]
            distance_diff -= distances[self.get_index(index, len(cycle))]
            distance_diff -= distances[self.get_index(other_index - 1, len(cycle))]
            distance_diff -= distances[self.get_index(other_index, len(cycle))]

            cycle[index], cycle[other_index] = cycle[other_index], cycle[index]

            distances[self.get_index(index - 1, len(cycle))] = dist_matrix[cycle[self.get_index(index - 1, len(cycle))], cycle[index]]
            distances[self.get_index(index, len(cycle))] = dist_matrix[cycle[index], cycle[self.get_index(index + 1, len(cycle))]]
            distances[self.get_index(other_index - 1, len(cycle))] = dist_matrix[cycle[self.get_index(other_index - 1, len(cycle))], cycle[other_index]]
            distances[self.get_index(other_index, len(cycle))] = dist_matrix[cycle[other_index], cycle[self.get_index(other_index + 1, len(cycle))]]

            distance_diff += distances[self.get_index(index - 1, len(cycle))]
            distance_diff += distances[self.get_index(index, len(cycle))]
            distance_diff += distances[self.get_index(other_index - 1, len(cycle))]
            distance_diff += distances[self.get_index(other_index, len(cycle))]
        else:
            # reverse
            i = index
            while self.get_index(i, len(cycle)) != other_index:
                distance_diff -= distances[self.get_index(i - 1, len(cycle))]
                i += 1
            distance_diff -= distances[self.get_index(other_index - 1, len(cycle))]
            distance_diff -= distances[self.get_index(other_index, len(cycle))]

            self.reverse(cycle, index, other_index)

            i = index
            while self.get_index(i, len(cycle)) != other_index:
                distances[self.get_index(i - 1, len(cycle))] = dist_matrix[cycle[self.get_index(i - 1, len(cycle))], cycle[self.get_index(i, len(cycle))]]
                i += 1
            distances[self.get_index(other_index - 1, len(cycle))] = dist_matrix[cycle[self.get_index(other_index - 1, len(cycle))], cycle[self.get_index(other_index, len(cycle))]]
            distances[self.get_index(other_index, len(cycle))] = dist_matrix[cycle[self.get_index(other_index, len(cycle))], cycle[self.get_index(other_index + 1, len(cycle))]]

            i = index
            while self.get_index(i, len(cycle)) != other_index:
                distance_diff += distances[self.get_index(i - 1, len(cycle))]
                i += 1
            distance_diff += distances[self.get_index(other_index - 1, len(cycle))]
            distance_diff += distances[self.get_index(other_index, len(cycle))]

        return distance_diff

if __name__ == "__main__":
    generator = GraphPlanarGenerator()

    n_operations_fn = lambda ns,nds,nd: 1
    accept_l_fn = lambda d,nd,s: 1.0
    accept_h_fn = lambda d,nd,s: np.random.binomial(n=1, p=np.exp((d - nd) / 1.))

    coords, dist_matrix = generator.generate()
    operation = Operation(dist_matrix.shape[0])

    tsp = TSP(
        dist_matrix,
        operation,
        n_operations_fn,
        accept_l_fn,
        accept_h_fn
    )
    try:
        greedy_distance, greedy_cycle, best_distance, best_cycle = tsp.solve()
        print("Кратчайшее расстояние из жадного алгоритма:", greedy_distance)
        print("Итоговое кратчайшее расстояние:", best_distance)

        visualize_cycles(coords, greedy_cycle, best_cycle)

    except NegativeCycleException as ex:
        print(ex)
    except UnreachableVertexException as ex:
        print(ex)
