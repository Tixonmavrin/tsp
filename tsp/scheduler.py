import numpy as np

class BaseScheduler:
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def __call__(self, distance, new_distance, step, losses, best_loss):
        tempr = min(0.01, 1. * pow(step + 1.0, -0.5))
        return np.random.binomial(n=1, p=np.exp((distance - new_distance) / tempr))