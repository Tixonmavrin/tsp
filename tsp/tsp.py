from scipy.sparse.csgraph import shortest_path, NegativeCycleError
import numpy as np
from typing import Union, Tuple

from .disjoint_set import DisjointSet
from .exceptions import NegativeCycleException, UnreachableVertexException
from .operation import SwapReverseOperation, SwapReverseDistanceOperation

ArrayLike = Union[float, list, np.ndarray]


def to_ndarray(data: ArrayLike) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.asarray(data)


def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]


class TSP:
    def __init__(
        self, graph, operation, n_operations_fn, accept_l_fn, accept_h_fn
    ):

        graph = to_ndarray(graph)
        if len(graph.shape) != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("Bad graph.")
        self.graph = graph

        self.operation = operation

        self.n_operations_fn = n_operations_fn
        self.accept_l_fn = accept_l_fn
        self.accept_h_fn = accept_h_fn

    def get_greedy_path_map(self, dist_matrix: np.ndarray) -> dict:
        positions = dist_matrix.ravel().argsort()
        pos_x = positions // dist_matrix.shape[0]
        pos_y = positions % dist_matrix.shape[0]

        from_set = set()
        to_set = set()
        disjoint_set = DisjointSet(dist_matrix.shape[0])

        path = dict()
        for x, y in zip(pos_x, pos_y):
            if x in from_set or y in to_set:
                continue
            if len(from_set) != dist_matrix.shape[0] - 1 and disjoint_set.connected(
                x, y
            ):
                continue

            from_set.add(x)
            to_set.add(y)
            disjoint_set.union(x, y)
            path[x] = y

            if len(from_set) == dist_matrix.shape[0]:
                break

        return path

    def get_cycle_distances(self, cycle: np.array, dist_matrix: np.ndarray) -> np.array:
        distances = []
        for i in range(len(cycle) - 1):
            distances.append(dist_matrix[cycle[i], cycle[i + 1]])
        distances.append(dist_matrix[cycle[len(cycle) - 1], cycle[0]])
        return np.array(distances)

    def greedy_solution(self, dist_matrix: np.ndarray) -> Tuple[np.array, np.array]:
        path = self.get_greedy_path_map(dist_matrix)

        cycle = []
        x = 0
        for _ in range(dist_matrix.shape[0]):
            x = path[x]
            cycle.append(x)
        cycle = np.asarray(cycle)

        distances = self.get_cycle_distances(cycle=cycle, dist_matrix=dist_matrix)
        return distances, cycle

    def improve_distance(
        self, dist_matrix: np.ndarray, distances: np.array, cycle: np.array, num_steps: int
    ) -> Tuple[np.array, np.array]:
        distance = np.sum(distances)
        best_distance = distance
        best_cycle = cycle
        losses = [distance]

        for step in range(num_steps):
            new_cycle = cycle.copy()
            new_distances = distances.copy()
            new_distance = distance

            n_operations = self.n_operations_fn(new_cycle, new_distances)
            for i in range(n_operations):
                distance_diff = self.operation(new_cycle, new_distances, dist_matrix)
                new_distance += distance_diff

            losses.append(new_distance)

            if new_distance < best_distance:
                best_distance = new_distance
                best_cycle = new_cycle

            if (
                new_distance < distance
                and self.accept_l_fn(distance, new_distance, step)
            ) or (
                new_distance >= distance
                and self.accept_h_fn(distance, new_distance, step)
            ):
                distances = new_distances
                distance = new_distance
                cycle = new_cycle

        return best_distance, best_cycle, losses

    def random_solution(self, dist_matrix: np.ndarray) -> Tuple[np.array, np.array]:
        cycle = np.arange(dist_matrix.shape[0])
        np.random.shuffle(cycle)

        distances = self.get_cycle_distances(cycle=cycle, dist_matrix=dist_matrix)
        return distances, cycle

    def expand_path(self, predecessors, cycle) -> np.array:
        cycle_ext = []
        for i in range(len(cycle) - 1):
            cycle_ext.extend(get_path(predecessors, cycle[i], cycle[i + 1])[1:])
        cycle_ext.extend(get_path(predecessors, cycle[len(cycle) - 1], cycle[0])[1:])
        cycle_ext = np.asarray(cycle_ext)
        return cycle_ext

    def solve(self, num_steps=100_000, use_greedy = True):
        try:
            dist_matrix, predecessors = shortest_path(
                csgraph=self.graph,
                method="auto",
                directed=True,
                return_predecessors=True,
                unweighted=False,
                overwrite=False,
            )
        except NegativeCycleError:
            raise NegativeCycleException("Found negative cycle.")

        if (dist_matrix == np.inf).any():
            raise UnreachableVertexException("All vertices must be reachable.")

        if use_greedy:
            start_distances, start_cycle = self.greedy_solution(dist_matrix=dist_matrix)
        else:
            start_distances, start_cycle = self.random_solution(dist_matrix=dist_matrix)
        start_cycle_ext = self.expand_path(
            predecessors=predecessors, cycle=start_cycle
        )
        best_distance, best_cycle, losses = self.improve_distance(
            dist_matrix=dist_matrix, distances=start_distances, cycle=start_cycle, num_steps=num_steps
        )
        best_cycle_ext = self.expand_path(predecessors=predecessors, cycle=best_cycle)

        return start_distances.sum(), start_cycle_ext, best_distance, best_cycle_ext, losses


class SwapReverseTSP(TSP):
    def __init__(
        self,
        graph,
        n_operations_fn,
        accept_l_fn,
        accept_h_fn,
        first_index_fn,
        delta_index_fn,
        p_fn,
    ):

        super().__init__(
            graph,
            SwapReverseOperation(
                first_index_fn=first_index_fn, delta_index_fn=delta_index_fn, p_fn=p_fn
            ),
            n_operations_fn,
            accept_l_fn,
            accept_h_fn
        )


class SwapReverseTSPFixed(SwapReverseTSP):
    def __init__(self, graph):

        super().__init__(
            graph=graph,
            n_operations_fn=lambda new_cycle, new_distances: np.random.randint(
                0, int(np.log(graph.shape[0])) + 1
            ),
            accept_l_fn=lambda distance, new_distance, step: 1.0,
            accept_h_fn=lambda distance, new_distance, step: 0.0,
            first_index_fn=lambda: np.random.randint(0, graph.shape[0]),
            delta_index_fn=lambda: np.random.randint(0, graph.shape[0]),
            p_fn=lambda: np.random.binomial(n=1, p=0.5),
        )


class SwapReverseTSPFixedAnnealing(SwapReverseTSP):
    def __init__(self, graph, theta=1.0):

        super().__init__(
            graph=graph,
            n_operations_fn=lambda new_cycle, new_distances: np.random.randint(
                0, int(np.log(graph.shape[0])) + 1
            ),
            accept_l_fn=lambda distance, new_distance, step: 1.0,
            accept_h_fn=lambda distance, new_distance, step: np.random.binomial(
                n=1, p=np.exp((distance - new_distance) / theta)
            ),
            first_index_fn=lambda: np.random.randint(0, graph.shape[0]),
            delta_index_fn=lambda: np.random.randint(0, graph.shape[0]),
            p_fn=lambda: np.random.binomial(n=1, p=0.5),
        )

class ThetaCounter:
    def __init__(self, theta, gamma):
        self.saved_theta = theta
        self.gamma = gamma
    def __call__(self):
        theta = self.saved_theta
        self.saved_theta *= self.gamma
        return theta

class SwapReverseTSPChangingExpAnnealing(SwapReverseTSP):
    def __init__(self, graph, theta=1., gamma=0.99):
        theta_fn = ThetaCounter(theta, gamma)
        super().__init__(
            graph=graph,
            n_operations_fn=lambda new_cycle, new_distances: np.random.randint(
                0, int(np.log(graph.shape[0])) + 1
            ),
            accept_l_fn=lambda distance, new_distance, step: 1.0,
            accept_h_fn=lambda distance, new_distance, step: np.random.binomial(
                n=1, p=np.exp((distance - new_distance) / theta_fn())
            ),
            first_index_fn=lambda: np.random.randint(0, graph.shape[0]),
            delta_index_fn=lambda: np.random.randint(0, graph.shape[0]),
            p_fn=lambda: np.random.binomial(n=1, p=0.5),
        )

class SwapReverseTSPChangingLinearAnnealing(SwapReverseTSP):
    def __init__(self, graph, theta_start=1., theta_end=1e-4, n_end=100_000):
        theta_fn = lambda step: step * (theta_end - theta_start) / n_end + theta_start if step <= n_end else theta_end
        super().__init__(
            graph=graph,
            n_operations_fn=lambda new_cycle, new_distances: np.random.randint(
                0, int(np.log(graph.shape[0])) + 1
            ),
            accept_l_fn=lambda distance, new_distance, step: 1.0,
            accept_h_fn=lambda distance, new_distance, step: np.random.binomial(
                n=1, p=np.exp((distance - new_distance) / theta_fn(step))
            ),
            first_index_fn=lambda: np.random.randint(0, graph.shape[0]),
            delta_index_fn=lambda: np.random.randint(0, graph.shape[0]),
            p_fn=lambda: np.random.binomial(n=1, p=0.5),
        )

class DefaultTheta:
    def __init__(self):
        pass
    def __call__(self, step):
        theta = min(0.1, 1. * pow(step + 1.0, -0.5))
        return theta

class SwapReverseTSPStable(TSP):
    def __init__(self, graph):
        theta_fn = DefaultTheta()
        p_fn=lambda: np.random.binomial(n=1, p=0.1)

        super().__init__(
            graph=graph,
            operation=SwapReverseDistanceOperation(
                p_fn=p_fn
            ),
            n_operations_fn=lambda new_cycle, new_distances: np.random.geometric(p=0.9),
            accept_l_fn=lambda distance, new_distance, step: 1.,
            accept_h_fn=lambda distance, new_distance, step: np.random.binomial(
                n=1, p=np.exp((distance - new_distance) / theta_fn(step))
            )
        )
