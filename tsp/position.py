import numpy as np

from .utils import stable_softmax


class SwapReverseRandomPositions:
    def __call__(self, operation, cycle, dist_matrix):
        return np.random.randint(0, len(cycle)), np.random.randint(0, len(cycle))

class SwapReverseSoftmaxPositions:
    def __init__(self, swap_diff_fn = lambda: np.random.geometric(p=0.5) * (np.random.binomial(n=1, p=0.5) * 2 - 1)):
        self.swap_diff_fn = swap_diff_fn

    def __call__(self, operation, cycle, dist_matrix):
        cycle_arange = np.arange(len(cycle))

        distances = np.array([dist_matrix[u, v] for u, v in zip(np.roll(cycle_arange, 1), cycle_arange)])
        distances_shifted = np.roll(distances, -1)
        sum_distances = distances + distances_shifted
        sum_distances_softmax = stable_softmax(sum_distances)

        if operation == "swap":
            first_index = np.random.choice(cycle_arange, p=sum_distances_softmax)
            diff = np.random.geometric(p=0.5) * (np.random.binomial(n=1, p=0.5) * 2 - 1)
            return first_index, first_index + diff

        elif operation == "reverse":
            first_index = np.random.choice(cycle_arange, p=sum_distances_softmax)
            sum_distances[first_index] = -np.inf
            sum_distances_softmax = stable_softmax(sum_distances)
            second_index = np.random.choice(cycle_arange, p=sum_distances_softmax)
            return first_index, second_index

        else:
            raise RuntimeError("Unknown operation.")
