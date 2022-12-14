from tsp import (
    TSP,
    NegativeCycleException,
    UnreachableVertexException,
    PlanarGraphGenerator,
    SimpleSwapReverseTSP,
    visualize_cycles,
)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n = 100
    generator = PlanarGraphGenerator(n=n)
    coords, dist_matrix = generator.generate()
    tsp = SimpleSwapReverseTSP(dist_matrix, num_steps=100_000)
    try:
        greedy_distance, greedy_cycle, best_distance, best_cycle = tsp.solve()
        print("Кратчайшее расстояние из жадного алгоритма:", greedy_distance)
        print("Итоговое кратчайшее расстояние:", best_distance)

        visualize_cycles(coords, greedy_cycle, best_cycle)

    except NegativeCycleException as ex:
        print(ex)
    except UnreachableVertexException as ex:
        print(ex)
