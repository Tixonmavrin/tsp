import numpy as np
import functools

class Indexer:
    def __init__(self, cycle, dist_matrix):
        self.cycle = cycle
        self.dist_matrix = dist_matrix

    def geti(self, i):
        return i % len(self.cycle)

    def geted(self, i):
        return self.dist_matrix[self.cycle[i], self.cycle[self.geti(i + 1)]]

    def getd(self, i):
        return self.geted(self.geti(i - 1)) + self.geted(i)

class SwapOperation:
    def __init__(self, indexes_fn):
        self.indexes_fn = indexes_fn

    def __call__(self, cycle, dist_matrix):
        indexer = Indexer(cycle, dist_matrix)

        first_index, second_index = self.indexes_fn(cycle, dist_matrix)
        first_index, second_index = indexer.geti(first_index), indexer.geti(second_index)

        if first_index == second_index:
            return 0

        if indexer.geti(first_index + 2) == first_index:
            cycle[first_index], cycle[second_index] = cycle[second_index], cycle[first_index]
            return 0

        distance_diff = 0

        if indexer.geti(first_index + 1) == second_index:
            distance_diff -= indexer.geted(indexer.geti(first_index - 1))
            distance_diff -= indexer.geted(first_index)
            distance_diff -= indexer.geted(second_index)

            cycle[first_index], cycle[second_index] = cycle[second_index], cycle[first_index]

            distance_diff += indexer.geted(indexer.geti(first_index - 1))
            distance_diff += indexer.geted(first_index)
            distance_diff += indexer.geted(second_index)

            return distance_diff

        if indexer.geti(second_index + 1) == first_index:
            distance_diff -= indexer.geted(indexer.geti(second_index - 1))
            distance_diff -= indexer.geted(second_index)
            distance_diff -= indexer.geted(first_index)

            cycle[first_index], cycle[second_index] = cycle[second_index], cycle[first_index]

            distance_diff += indexer.geted(indexer.geti(second_index - 1))
            distance_diff += indexer.geted(second_index)
            distance_diff += indexer.geted(first_index)

            return distance_diff

        distance_diff -= indexer.getd(first_index)
        distance_diff -= indexer.getd(second_index)

        cycle[first_index], cycle[second_index] = cycle[second_index], cycle[first_index]

        distance_diff += indexer.getd(first_index)
        distance_diff += indexer.getd(second_index)

        return distance_diff

class ReverseOperation:
    def __init__(self, indexes_fn):
        self.indexes_fn = indexes_fn

    def reverse(self, indexer, cycle, first_index, second_index):
        while first_index != second_index and first_index != indexer.geti(second_index - 1):
            cycle[first_index], cycle[second_index] = cycle[second_index], cycle[first_index]
            first_index = indexer.geti(first_index + 1)
            second_index = indexer.geti(second_index - 1)

        if first_index == indexer.geti(second_index - 1):
            cycle[first_index], cycle[second_index] = cycle[second_index], cycle[first_index]

    def __call__(self, cycle, dist_matrix):
        indexer = Indexer(cycle, dist_matrix)

        first_index, second_index = self.indexes_fn(cycle, dist_matrix)
        first_index, second_index = indexer.geti(first_index), indexer.geti(second_index)

        if first_index == second_index:
            return 0

        if indexer.geti(first_index + 2) == first_index:
            cycle[first_index], cycle[second_index] = cycle[second_index], cycle[first_index]
            return 0

        distance_diff = 0

        if indexer.geti(second_index + 1) == first_index:
            i = first_index
            while indexer.geti(i) != second_index:
                distance_diff -= indexer.geted(indexer.geti(i - 1))
                i += 1
            distance_diff -= indexer.geted(indexer.geti(second_index - 1))

            self.reverse(indexer, cycle, first_index, second_index)

            i = first_index
            while indexer.geti(i) != second_index:
                distance_diff += indexer.geted(indexer.geti(i - 1))
                i += 1
            distance_diff += indexer.geted(indexer.geti(second_index - 1))

            return distance_diff

        i = first_index
        while indexer.geti(i) != second_index:
            distance_diff -= indexer.geted(indexer.geti(i - 1))
            i += 1
        distance_diff -= indexer.getd(second_index)

        self.reverse(indexer, cycle, first_index, second_index)

        i = first_index
        while indexer.geti(i) != second_index:
            distance_diff += indexer.geted(indexer.geti(i - 1))
            i += 1
        distance_diff += indexer.getd(second_index)

        return distance_diff

class SwapReverseOperation:
    def __init__(self, indexes_fn, p):
        self.reverse_operation = ReverseOperation(functools.partial(indexes_fn, "reverse"))
        self.swap_operation = SwapOperation(functools.partial(indexes_fn, "swap"))

        self.p = p

    def __call__(self, cycle, dist_matrix):
        if np.random.binomial(n=1, p=self.p):
            return self.reverse_operation(cycle, dist_matrix)
        return self.swap_operation(cycle, dist_matrix)
