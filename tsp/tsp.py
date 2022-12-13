from scipy.sparse.csgraph import shortest_path, NegativeCycleError
import numpy as np
from typing import Union, Tuple

from .disjoint_set import DisjointSet
from .exceptions import NegativeCycleException, UnreachableVertexException

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

def get_index(index, length):
    result = index % length
    if result < 0 or result >= length:
        raise RuntimeError("Bad getting index. If you see this error, please open issue.")
    return result

class TSP:
    def __init__(self, graph, num_transpose_generator = None, index_generator = None, delta_index_generator = None, temperature = 1., num_steps = 100000):
        graph = to_ndarray(graph)
        if len(graph.shape) != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("Bad graph.")
        self.graph = graph

        if num_transpose_generator is None:
            num_transpose_generator = lambda: 1
        self.num_transpose_generator = num_transpose_generator

        if index_generator is None:
            index_generator = lambda: np.random.randint(0, graph.shape[0])
        self.index_generator = index_generator

        if delta_index_generator is None:
            delta_index_generator = lambda: np.random.randint(0, graph.shape[0])
        self.delta_index_generator = delta_index_generator

        self.temperature = temperature
        self.num_steps = num_steps

    def greedy_solution(self, dist_matrix: np.ndarray) -> Tuple[float, np.array]:
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
            if len(from_set) != len(to_set):
                raise RuntimeError("Bad greedy solution. If you see this error, please open issue.")
            if len(from_set) != dist_matrix.shape[0] - 1 and disjoint_set.connected(x, y):
                continue
            from_set.add(x)
            to_set.add(y)
            disjoint_set.union(x, y)
            path[x] = y

        cycle = []
        x = 0
        for _ in range(dist_matrix.shape[0]):
            x = path[x]
            cycle.append(x)
        cycle = np.asarray(cycle)

        distance = 0
        for i in range(len(cycle) - 1):
            distance += dist_matrix[cycle[i], cycle[i + 1]]
        distance += dist_matrix[cycle[len(cycle) - 1], cycle[0]]

        if x != 0 or len(cycle) != dist_matrix.shape[0]:
            raise RuntimeError("Bad cycle. If you see this error, please open issue.")

        return distance, cycle
    
    def improve_distance(self, dist_matrix: np.ndarray, distance: float, cycle: np.array) -> Tuple[float, np.array]:
        best_distance = distance
        best_cycle = cycle
        for step in range(self.num_steps):
            cycle_copy = cycle.copy()
            new_distance = distance

            num_transpose = self.num_transpose_generator()
            for transpose in range(num_transpose):
                index = get_index(self.index_generator(), len(cycle))
                delta = self.delta_index_generator()
                other_index = get_index(index + delta, len(cycle))

                new_distance -= dist_matrix[cycle_copy[get_index(index - 1, len(cycle))], cycle_copy[index]]
                new_distance -= dist_matrix[cycle_copy[index], cycle_copy[get_index(index + 1, len(cycle))]]
                new_distance -= dist_matrix[cycle_copy[get_index(other_index - 1, len(cycle))], cycle_copy[other_index]]
                new_distance -= dist_matrix[cycle_copy[other_index], cycle_copy[get_index(other_index + 1, len(cycle))]]

                cycle_copy[index], cycle_copy[other_index] = cycle_copy[other_index], cycle_copy[index]

                new_distance += dist_matrix[cycle_copy[get_index(index - 1, len(cycle))], cycle_copy[index]]
                new_distance += dist_matrix[cycle_copy[index], cycle_copy[get_index(index + 1, len(cycle))]]
                new_distance += dist_matrix[cycle_copy[get_index(other_index - 1, len(cycle))], cycle_copy[other_index]]
                new_distance += dist_matrix[cycle_copy[other_index], cycle_copy[get_index(other_index + 1, len(cycle))]]

            if new_distance < best_distance:
                best_distance = new_distance
                best_cycle = cycle_copy

            if new_distance < distance:
                distance = new_distance
                cycle = cycle_copy
            elif np.random.binomial(n=1, p=np.exp((distance - new_distance) / self.temperature)):
                distance = new_distance
                cycle = cycle_copy

        return best_distance, best_cycle

    def expand_path(self, dist_matrix, predecessors, best_cycle) -> Tuple[float, np.array]:
        best_distance = 0
        best_cycle_ext = []
        for i in range(len(best_cycle) - 1):
            best_distance += dist_matrix[best_cycle[i], best_cycle[i + 1]]
            best_cycle_ext.extend(get_path(predecessors, best_cycle[i], best_cycle[i + 1])[1:])
        best_distance += dist_matrix[best_cycle[len(best_cycle) - 1], best_cycle[0]]
        best_cycle_ext.extend(get_path(predecessors, best_cycle[len(best_cycle) - 1], best_cycle[0])[1:])
        best_cycle_ext = np.asarray(best_cycle_ext)

        if (best_cycle_ext - np.roll(best_cycle_ext, -1) == 0).any():
            raise RuntimeError("Bad cycle. If you see this error, please open issue.")
        flags = np.arange(dist_matrix.shape[0])
        for v in best_cycle_ext:
            flags[v] = 1
        if (flags != 1).any():
            raise RuntimeError("Bad cycle. If you see this error, please open issue.")

        return best_distance, best_cycle_ext

    def solve(self):
        try:
            dist_matrix, predecessors = shortest_path(
                csgraph=self.graph, 
                method="auto",
                directed=True, 
                return_predecessors=True, 
                unweighted=False, 
                overwrite=False
            )
        except NegativeCycleError:
            raise NegativeCycleException("Found negative cycle.")

        if (dist_matrix == np.inf).any():
            raise UnreachableVertexException("All vertices must be reachable.")

        greedy_distance, greedy_cycle = self.greedy_solution(dist_matrix=dist_matrix)
        greedy_distance_ext, greedy_cycle_ext = self.expand_path(dist_matrix=dist_matrix, predecessors=predecessors, best_cycle=greedy_cycle)
        best_distance, best_cycle = self.improve_distance(dist_matrix=dist_matrix, distance=greedy_distance, cycle=greedy_cycle)
        best_distance_ext, best_cycle_ext = self.expand_path(dist_matrix=dist_matrix, predecessors=predecessors, best_cycle=best_cycle)

        if (not np.isclose(best_distance_ext, best_distance)):
            raise RuntimeError("Bad best distance ext. If you see this error, please open issue.")
        return greedy_distance_ext, greedy_cycle_ext, best_distance_ext, np.asarray(best_cycle_ext)
