from tsp import (
    TSP,
    NegativeCycleException,
    UnreachableVertexException,
    PlanarGraphGenerator,
    SwapReverseTSPStable,
    visualize_cycles,
    visualize_losses,
)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n = 293
    generator = PlanarGraphGenerator(n=n)
    coords, dist_matrix = generator.generate()
    tsp = SwapReverseTSPStable(dist_matrix)
    try:
        greedy_distance, greedy_cycle, best_distance, best_cycle, losses = tsp.solve(num_steps=100_000, use_greedy=True)
        print("Кратчайшее расстояние из жадного алгоритма:", greedy_distance)
        print("Итоговое кратчайшее расстояние:", best_distance)

        visualize_cycles(coords, greedy_cycle, best_cycle)
        visualize_losses(losses)

    except NegativeCycleException as ex:
        print(ex)
    except UnreachableVertexException as ex:
        print(ex)
