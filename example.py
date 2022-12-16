from tsp import (
    SwapReverseTSPStable,
    NegativeCycleException,
    UnreachableVertexException,
    PlanarGraphGeneratorUniform,
    visualize_cycles,
    visualize_losses,
)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    n = 300
    generator = PlanarGraphGeneratorUniform(n=n)
    coords, dist_matrix = generator.generate()
    tsp = SwapReverseTSPStable(dist_matrix, num_steps=100_000, use_greedy=False)
    try:
        start_time = time.time()
        greedy_distance, greedy_cycle, best_distance, best_cycle, losses = tsp.solve()
        end_time = time.time()
        print("Кратчайшее расстояние из первого приближения:", greedy_distance)
        print("Итоговое кратчайшее расстояние:", best_distance)
        print("Время работы:", end_time - start_time)

        visualize_cycles(coords, greedy_cycle, best_cycle)
        visualize_losses(losses)

    except NegativeCycleException as ex:
        print(ex)
    except UnreachableVertexException as ex:
        print(ex)
