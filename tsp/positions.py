import numpy as np

def stable_softmax(x):
    z = x - np.max(x)
    z = z - np.log(np.sum(np.exp(z)))
    return np.exp(z)

class RandomPositions:
    def __init__(self):
        pass
    
    def __call__(self, operation, cycle, dist_matrix):
        return np.random.randint(0, len(cycle)), np.random.randint(0, len(cycle))

class SoftmaxPositions:
    def __init__(self):
        pass
    
    def __call__(self, operation, cycle, dist_matrix):
        cycle_arange = np.arange(len(cycle))

        distances = np.array([dist_matrix[u, v] for u, v in zip(cycle_arange, np.roll(cycle_arange, 1))])
        distances_shifted = np.roll(distances, 1)
        sum_distances = distances + distances_shifted
        sum_distances_softmax = stable_softmax(sum_distances)

        if operation == "swap":
            first_index = np.random.choice(cycle_arange, p=sum_distances_softmax)
            diff = np.random.geometric(p=0.5) * (np.random.binomial(n=1, p=0.5) * 2 - 1)
            return first_index, first_index + diff

        elif operation == "reverse":
            first_index = np.random.choice(cycle_arange, p=sum_distances_softmax)
            sum_distances[first_index] = .0
            sum_distances_softmax = stable_softmax(sum_distances)
            second_index = np.random.choice(cycle_arange, p=sum_distances_softmax)
            return first_index, second_index

        else:
            raise RuntimeError("Unknown operation.")
