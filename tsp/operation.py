import numpy as np

class SwapReverseOperation:
    def __init__(self, indexes_fn, p):
        self.indexes_fn = indexes_fn
        self.p = p

        self.length = None
        self.cycle = None
        self.dist_matrix = None

    def geti(self, i):
        return i % self.length

    def getnd(self, i):
        return self.dist_matrix[self.cycle[self.geti(i)], self.cycle[self.geti(i + 1)]]

    def getd(self, i):
        return self.getnd(i - 1) + self.getnd(i)

    def reverse(self, cycle, first_index, second_index):
        while first_index != second_index and first_index != self.geti(second_index - 1):
            cycle[first_index], cycle[second_index] = cycle[second_index], cycle[first_index]
            first_index = self.geti(first_index + 1)
            second_index = self.geti(second_index - 1)

        if first_index == self.geti(second_index - 1):
            cycle[first_index], cycle[second_index] = cycle[second_index], cycle[first_index]

    def __call__(self, cycle, dist_matrix):
        self.length = len(cycle)
        self.cycle = cycle
        self.dist_matrix = dist_matrix

        distance_diff = 0
        if np.random.binomial(n=1, p=self.p):
            first_index, second_index = self.indexes_fn("reverse", cycle, dist_matrix)
            first_index, second_index = self.geti(first_index), self.geti(second_index)

            i = first_index
            while self.geti(i) != second_index:
                distance_diff -= self.getnd(i - 1)
                i += 1
            distance_diff -= self.getd(second_index)

            self.reverse(cycle, first_index, second_index)

            i = first_index
            while self.geti(i) != second_index:
                distance_diff += self.getnd(i - 1)
                i += 1
            distance_diff += self.getd(second_index)

        else:
            first_index, second_index = self.indexes_fn("swap", cycle, dist_matrix)
            first_index, second_index = self.geti(first_index), self.geti(second_index)

            distance_diff -= self.getd(first_index)
            distance_diff -= self.getd(second_index)

            cycle[first_index], cycle[second_index] = cycle[second_index], cycle[first_index]

            distance_diff += self.getd(first_index)
            distance_diff += self.getd(second_index)

        return distance_diff
