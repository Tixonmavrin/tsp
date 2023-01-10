from scipy.sparse.csgraph import shortest_path, NegativeCycleError
import numpy as np
from typing import Union, Tuple

from .utils import DisjointSet
from .exceptions import NegativeCycleException, UnreachableVertexException
from .position import SwapReverseRandomPositions, SwapReverseSoftmaxPositions
from .operation import SwapReverseOperation
from .scheduler import BaseScheduler

ArrayLike = Union[float, list, np.ndarray]
def to_ndarray(data: ArrayLike) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.asarray(data)

class GraphUtils:
    @staticmethod
    def reconstruct_graph_path(Pr, i, j):
        path = [j]
        k = j
        while Pr[i, k] != -9999:
            path.append(Pr[i, k])
            k = Pr[i, k]
        return path[::-1]

    @staticmethod
    def cycle_distances(cycle: np.array, dist_matrix: np.ndarray) -> np.array:
        distances = []
        for i in range(len(cycle) - 1):
            distances.append(dist_matrix[cycle[i], cycle[i + 1]])
        distances.append(dist_matrix[cycle[len(cycle) - 1], cycle[0]])
        return np.array(distances)

    @staticmethod
    def expand_path(predecessors, cycle) -> np.array:
        cycle_ext = []
        for i in range(len(cycle) - 1):
            cycle_ext.extend(GraphUtils.reconstruct_graph_path(predecessors, cycle[i], cycle[i + 1])[1:])
        cycle_ext.extend(GraphUtils.reconstruct_graph_path(predecessors, cycle[len(cycle) - 1], cycle[0])[1:])
        cycle_ext = np.asarray(cycle_ext)
        return cycle_ext

class GreedySolution:
    def __init__(self, dist_matrix):
        self.dist_matrix = dist_matrix

    def get_greedy_path_map(self) -> dict:
        positions = self.dist_matrix.ravel().argsort()
        pos_x = positions // self.dist_matrix.shape[0]
        pos_y = positions % self.dist_matrix.shape[0]

        from_set = set()
        to_set = set()
        disjoint_set = DisjointSet(self.dist_matrix.shape[0])

        path = dict()
        for x, y in zip(pos_x, pos_y):
            if x in from_set or y in to_set:
                continue
            if len(from_set) != self.dist_matrix.shape[0] - 1 and disjoint_set.connected(
                x, y
            ):
                continue

            from_set.add(x)
            to_set.add(y)
            disjoint_set.union(x, y)
            path[x] = y

            if len(from_set) == self.dist_matrix.shape[0]:
                break

        return path

    def __call__(self):
        path = self.get_greedy_path_map()

        cycle = []
        x = 0
        for _ in range(self.dist_matrix.shape[0]):
            x = path[x]
            cycle.append(x)
        cycle = np.asarray(cycle)
        return cycle

class RandomSolution:
    def __init__(self, length):
        self.length = length

    def __call__(self):
        cycle = np.arange(self.length)
        np.random.shuffle(cycle)
        return cycle


class TSP:
    def __init__(
        self, graph, operation, n_operations_fn, accept_l_fn, accept_h_fn, num_steps=100_000, use_greedy = True
    ):

        graph = to_ndarray(graph)
        if len(graph.shape) != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("Bad graph.")
        self.graph = graph

        self.operation = operation

        self.n_operations_fn = n_operations_fn
        self.accept_l_fn = accept_l_fn
        self.accept_h_fn = accept_h_fn

        self.num_steps = num_steps
        self.use_greedy = use_greedy

    def improve_distance(
        self, dist_matrix: np.ndarray, cycle: np.array
    ) -> Tuple[np.array, np.array]:
        distance = np.sum(GraphUtils.cycle_distances(cycle=cycle, dist_matrix=dist_matrix))
        start_distance = distance
        best_distance = distance

        best_cycle = cycle
        losses = [distance]

        for step in range(self.num_steps):
            new_cycle = cycle.copy()
            new_distance = distance

            n_operations = self.n_operations_fn(new_cycle, dist_matrix, step, losses)
            for i in range(n_operations):
                distance_diff = self.operation(new_cycle, dist_matrix)
                new_distance += distance_diff

            losses.append(new_distance)

            if new_distance < best_distance:
                best_distance = new_distance
                best_cycle = new_cycle

            if (
                new_distance < distance
                and self.accept_l_fn(distance, new_distance, step, losses, best_distance)
            ) or (
                new_distance >= distance
                and self.accept_h_fn(distance, new_distance, step, losses, best_distance)
            ):
                distance = new_distance
                cycle = new_cycle

        return start_distance, best_distance, best_cycle, losses

    def solve(self):
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

        if dist_matrix.shape[0] == 1:
            return 0, np.array([0]), 0, np.array([0]), []

        if self.use_greedy:
            solution = GreedySolution(dist_matrix)
        else:
            solution = RandomSolution(dist_matrix.shape[0])
        start_cycle = solution()

        start_cycle_ext = GraphUtils.expand_path(
            predecessors=predecessors, cycle=start_cycle
        )
        start_distance, best_distance, best_cycle, losses = self.improve_distance(
            dist_matrix=dist_matrix, cycle=start_cycle
        )
        best_cycle_ext = GraphUtils.expand_path(predecessors=predecessors, cycle=best_cycle)

        return start_distance, start_cycle_ext, best_distance, best_cycle_ext, losses

class SwapReverseTSP(TSP):
    def __init__(
        self,
        graph,
        num_steps = 100_000,
        use_greedy = True,
        use_softmax = True,
        n_operations_p = 0.95,
        swap_reverse_p = 0.9
    ):
        if use_softmax:
            positions_fn = SwapReverseSoftmaxPositions()
        else:
            positions_fn = SwapReverseRandomPositions()

        scheduler = BaseScheduler(num_steps)
        n_operations_fn = lambda cycle, dist_matrix, step, losses: np.random.geometric(p=n_operations_p)

        operation = SwapReverseOperation(positions_fn, swap_reverse_p)
        accept_l_fn = lambda distance, new_distance, step, losses, best_loss: True

        super().__init__(
            graph=graph,
            operation=operation,
            n_operations_fn=n_operations_fn,
            accept_l_fn=accept_l_fn,
            accept_h_fn=scheduler,
            num_steps=num_steps, 
            use_greedy=use_greedy
        )
