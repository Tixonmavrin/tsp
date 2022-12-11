from tsp import TSP, NegativeCycleException, UnreachableVertexException
import numpy as np

class GraphNormalGenerator:
    def __init__(self, n = 1000, loc = 50.0, scale = 13.25, ninf = None):
        self.n = n
        self.loc = loc
        self.scale = scale
        if ninf is None:
            ninf = int(np.sqrt(self.n))
        self.ninf = ninf

    def generate(self):
        matrix = np.random.normal(loc=self.loc, scale=self.scale, size=(self.n, self.n))
        for _ in range(self.ninf):
            index = np.random.randint(0, self.n, size=2)
            matrix[index[0], index[1]] = np.inf
        return matrix


if __name__ == "__main__":
    generator = GraphNormalGenerator()
    n_experiments = 100
    for i in range(n_experiments):
        tsp = TSP(generator.generate())
        try:
            greedy_distance, shortest_distance, shortest_cycle = tsp.solve()
            print("Пример", i)
            print("Кратчайшее расстояние из жадного алгоритма:", greedy_distance)
            print("Итоговое кратчайшее расстояние:", shortest_distance)
            print("Итоговая длина цикла:", len(shortest_cycle))
            print("Итоговый цикл:", shortest_cycle)
            print()

        except NegativeCycleException as ex:
            print(ex)
        except UnreachableVertexException as ex:
            print(ex)
