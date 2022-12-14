import numpy as np


class SwapReverseOperation:
    def __init__(self, first_index_fn, delta_index_fn, p_fn):
        self.first_index_fn = first_index_fn
        self.delta_index_fn = delta_index_fn

        self.p_fn = p_fn

    def get_index(self, i, length):
        return i % length

    def reverse(self, cycle, index, other_index):
        while index != other_index and index != self.get_index(
            other_index - 1, len(cycle)
        ):
            cycle[index], cycle[other_index] = cycle[other_index], cycle[index]
            index = self.get_index(index + 1, len(cycle))
            other_index = self.get_index(other_index - 1, len(cycle))

        if index == self.get_index(other_index - 1, len(cycle)):
            cycle[index], cycle[other_index] = cycle[other_index], cycle[index]

    def __call__(self, cycle, distances, dist_matrix):
        index = self.get_index(self.first_index_fn(), len(cycle))
        delta = self.delta_index_fn()
        other_index = self.get_index(index + delta, len(cycle))
        distance_diff = 0

        if self.p_fn():
            # Swap
            distance_diff -= distances[self.get_index(index - 1, len(cycle))]
            distance_diff -= distances[self.get_index(index, len(cycle))]
            distance_diff -= distances[self.get_index(other_index - 1, len(cycle))]
            distance_diff -= distances[self.get_index(other_index, len(cycle))]

            cycle[index], cycle[other_index] = cycle[other_index], cycle[index]

            distances[self.get_index(index - 1, len(cycle))] = dist_matrix[
                cycle[self.get_index(index - 1, len(cycle))], cycle[index]
            ]
            distances[self.get_index(index, len(cycle))] = dist_matrix[
                cycle[index], cycle[self.get_index(index + 1, len(cycle))]
            ]
            distances[self.get_index(other_index - 1, len(cycle))] = dist_matrix[
                cycle[self.get_index(other_index - 1, len(cycle))], cycle[other_index]
            ]
            distances[self.get_index(other_index, len(cycle))] = dist_matrix[
                cycle[other_index], cycle[self.get_index(other_index + 1, len(cycle))]
            ]

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
                distances[self.get_index(i - 1, len(cycle))] = dist_matrix[
                    cycle[self.get_index(i - 1, len(cycle))],
                    cycle[self.get_index(i, len(cycle))],
                ]
                i += 1
            distances[self.get_index(other_index - 1, len(cycle))] = dist_matrix[
                cycle[self.get_index(other_index - 1, len(cycle))],
                cycle[self.get_index(other_index, len(cycle))],
            ]
            distances[self.get_index(other_index, len(cycle))] = dist_matrix[
                cycle[self.get_index(other_index, len(cycle))],
                cycle[self.get_index(other_index + 1, len(cycle))],
            ]

            i = index
            while self.get_index(i, len(cycle)) != other_index:
                distance_diff += distances[self.get_index(i - 1, len(cycle))]
                i += 1
            distance_diff += distances[self.get_index(other_index - 1, len(cycle))]
            distance_diff += distances[self.get_index(other_index, len(cycle))]

        return distance_diff
