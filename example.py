from tsp import (
    SwapReverseTSP,
    NegativeCycleException,
    UnreachableVertexException,
    PlanarGraphGeneratorUniform,
    visualize_cycles,
    visualize_losses,
)
import time

if __name__ == "__main__":
    n = 300
    num_steps = 200_000
    use_greedy = True
    use_softmax = True

    generator = PlanarGraphGeneratorUniform(n=n)
    coords, dist_matrix = generator.generate()
    tsp = SwapReverseTSP(dist_matrix, num_steps=num_steps, use_greedy=use_greedy, use_softmax=use_softmax)
    try:
        start_time = time.time()
        base_distance, base_cycle, best_distance, best_cycle, losses = tsp.solve()
        end_time = time.time()

        print("Кратчайшее расстояние из первого приближения:", base_distance)
        print("Итоговое кратчайшее расстояние:", best_distance)
        print("Время работы:", end_time - start_time)

        visualize_cycles(coords, base_cycle, best_cycle)
        visualize_losses(losses)

    except NegativeCycleException as ex:
        print(ex)
    except UnreachableVertexException as ex:
        print(ex)
