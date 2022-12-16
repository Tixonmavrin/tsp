import numpy as np

class PlanarGraphGeneratorNormal:
    def __init__(self, n=20, loc=0.0, scale=1.0):
        self.n = n
        self.loc = loc
        self.scale = scale

    def generate(self):
        coords = np.random.normal(loc=self.loc, scale=self.scale, size=(self.n, 2))
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        return coords, dist_matrix

class PlanarGraphGeneratorUniform:
    def __init__(self, n=20, low=0.0, high=1.0):
        self.n = n
        self.low = low
        self.high = high

    def generate(self):
        coords = np.random.uniform(low=self.low, high=self.high, size=(self.n, 2))
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        return coords, dist_matrix
