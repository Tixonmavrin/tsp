import numpy as np
from scipy.spatial import distance_matrix

class PlanarGraphGeneratorNormal:
    def __init__(self, n=20, loc=0.0, scale=1.0):
        self.n = n
        self.loc = loc
        self.scale = scale

    def generate(self):
        coords = np.random.normal(loc=self.loc, scale=self.scale, size=(self.n, 2))
        return coords, distance_matrix(coords, coords)

class PlanarGraphGeneratorUniform:
    def __init__(self, n=20, low=0.0, high=1.0):
        self.n = n
        self.low = low
        self.high = high

    def generate(self):
        coords = np.random.uniform(low=self.low, high=self.high, size=(self.n, 2))
        return coords, distance_matrix(coords, coords)

class RandomGraphGeneratorNormal:
    def __init__(self, n=20, loc=1.0, scale=2.0):
        self.n = n
        self.loc = loc
        self.scale = scale

    def generate(self):
        dist_matrix = np.random.normal(loc=self.loc, scale=self.scale, size=(self.n, self.n))
        return dist_matrix
