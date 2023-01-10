import numpy as np


class DisjointSet:
    def __init__(self, n: int) -> None:
        self._data: np.ndarray = np.arange(n)

    def find(self, x: int) -> int:
        while x != self._data[x]:
            self._data[x] = self._data[self._data[x]]
            x = self._data[x]
        return x

    def union(self, x: int, y: int) -> None:
        parent_x, parent_y = self.find(x), self.find(y)
        if parent_x != parent_y:
            self._data[parent_x] = parent_y

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

def stable_softmax(x):
    z = x - np.max(x)
    z = z - np.log(np.sum(np.exp(z)))
    return np.exp(z)