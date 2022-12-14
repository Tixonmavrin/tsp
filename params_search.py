from tsp import TSP, NegativeCycleException, UnreachableVertexException
import numpy as np


class GraphPlanarGenerator:
    def __init__(self, n = 20, loc = 0.0, scale = 1.0):
        self.n = n
        self.loc = loc
        self.scale = scale

    def generate(self):
        coords = np.random.normal(loc=self.loc, scale=self.scale, size=(self.n, 2))
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        return coords, dist_matrix

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
    count_per_n = 5
    for n in [10, 20, 50, 100, 200, 500, 1000]:
        for generator in [GraphPlanarGenerator(n)]:
            for i in range(count_per_n):
                coords, dist_matrix = generator.generate()
                ###
                for first_index_fn in [
                    lambda: np.random.randint(0, n),
                    lambda: np.random.geometric(0.1, size=1),
                    lambda: np.random.geometric(0.3, size=1),
                    lambda: np.random.geometric(0.5, size=1),
                    lambda: np.random.geometric(0.7, size=1),
                    lambda: np.random.geometric(0.9, size=1),
                ]:
                    for delta_index_fn in [
                        lambda: np.random.randint(0, n),
                        lambda: np.random.geometric(0.1, size=1),
                        lambda: np.random.geometric(0.3, size=1),
                        lambda: np.random.geometric(0.5, size=1),
                        lambda: np.random.geometric(0.7, size=1),
                        lambda: np.random.geometric(0.9, size=1),
                    ]:
                        for p_generator in [
                            np.random.binomial(n=1, p=0.0),
                            np.random.binomial(n=1, p=0.3),
                            np.random.binomial(n=1, p=0.5),
                            np.random.binomial(n=1, p=0.7),
                            np.random.binomial(n=1, p=1.0),
                        ]:
                            for operation in [Operation(n, first_index_fn, delta_index_fn, p_generator)]:
                                for n_operations_fn in [
                                    lambda ns,nds,nd: 1,
                                    lambda ns,nds,nd: 10,
                                    lambda: np.random.geometric(0.1, size=1),
                                    lambda: np.random.geometric(0.3, size=1),
                                    lambda: np.random.geometric(0.5, size=1),
                                    lambda: np.random.geometric(0.7, size=1),
                                    lambda: np.random.geometric(0.9, size=1),
                                ]:
                                    for accept_l_fn in [
                                        lambda d,nd,s: np.random.binomial(n=1, p=1.0),
                                        lambda d,nd,s: np.random.binomial(n=1, p=0.9),
                                        lambda d,nd,s: np.random.binomial(n=1, p=0.5),
                                        lambda d,nd,s: nd / d < 0.95,
                                        lambda d,nd,s: np.random.binomial(n=1, p=np.exp((nd - d) / 1.)),
                                        lambda d,nd,s: np.random.binomial(n=1, p=np.exp((nd - d) / 2.)),
                                        lambda d,nd,s: np.random.binomial(n=1, p=np.exp((nd - d) / .5))
                                    ]:
                                        for accept_h_fn in [
                                            lambda d,nd,s: np.random.binomial(n=1, p=np.exp((d - nd) / 1.)),
                                            lambda d,nd,s: np.random.binomial(n=1, p=np.exp((d - nd) / 2.)),
                                            lambda d,nd,s: np.random.binomial(n=1, p=np.exp((d - nd) / .5)),
                                            lambda d,nd,s: np.random.binomial(n=1, p=1.0),
                                            lambda d,nd,s: np.random.binomial(n=1, p=0.9),
                                            lambda d,nd,s: np.random.binomial(n=1, p=0.5),
                                            lambda d,nd,s: d / nd > 0.95,
                                            lambda d,nd,s: np.random.binomial(n=1, p=np.exp((d - nd) / (1. * np.log(2 + s)))),
                                            lambda d,nd,s: np.random.binomial(n=1, p=np.exp((d - nd) / (2. * np.log(2 + s)))),
                                            lambda d,nd,s: np.random.binomial(n=1, p=np.exp((d - nd) / (.5 * np.log(2 + s)))),
                                        ]:
                                            for num_steps in [100, 1000, 10_000, 100_000]:

                                                tsp = TSP(
                                                    dist_matrix,
                                                    operation,
                                                    n_operations_fn,
                                                    accept_l_fn,
                                                    accept_h_fn,
                                                    num_steps=num_steps
                                                )

                                                try:
                                                    greedy_distance, greedy_cycle, best_distance, best_cycle = tsp.solve()
                                                    print(greedy_distance, best_distance)

                                                except NegativeCycleException as ex:
                                                    print(ex)
                                                except UnreachableVertexException as ex:
                                                    print(ex)
                ###