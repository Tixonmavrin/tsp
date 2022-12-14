import numpy as np


class PlanarGraphGenerator:
    def __init__(self, n=20, loc=0.0, scale=1.0):
        self.n = n
        self.loc = loc
        self.scale = scale

    def generate(self):
        coords = np.random.normal(loc=self.loc, scale=self.scale, size=(self.n, 2))
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        return coords, dist_matrix
