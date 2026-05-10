"""
Baseline Algorithms for HAQOA Comparison
Implements: GA, SA, PSO (adapted), ACO
All return a dict with keys: best_quality, best_solution, history, elapsed_seconds
"""

from __future__ import annotations

import time
import math
import random
from typing import List, Dict, Any

import numpy as np

from haqoa.problems.tsp import TSPInstance


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _run_result(best_sol, best_q, history, elapsed) -> Dict[str, Any]:
    return {
        "best_solution": list(best_sol),
        "best_quality": best_q,
        "history": history,          # list of best_quality per iteration
        "elapsed_seconds": elapsed,
    }


# ─── Genetic Algorithm ────────────────────────────────────────────────────────

class GeneticAlgorithm:
    """
    Standard GA with tournament selection, OX crossover, swap mutation.
    """

    def __init__(
        self,
        tsp: TSPInstance,
        population_size: int = 50,
        max_generations: int = 500,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        tournament_size: int = 5,
        elite_size: int = 5,
        seed: int = 42,
    ):
        self.tsp = tsp
        self.pop_size = population_size
        self.max_gen = max_generations
        self.mut_rate = mutation_rate
        self.cx_rate = crossover_rate
        self.tourn_k = tournament_size
        self.elite_size = elite_size
        random.seed(seed)
        np.random.seed(seed)

    def _random_route(self):
        r = list(range(self.tsp.n))
        random.shuffle(r)
        return r

    def _fitness(self, route):
        return self.tsp.route_distance(route)

    def _tournament(self, pop, fits):
        candidates = random.sample(range(len(pop)), self.tourn_k)
        return pop[min(candidates, key=lambda i: fits[i])]

    def _ox(self, p1, p2):
        n = len(p1)
        s, e = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[s:e + 1] = p1[s:e + 1]
        fill = [x for x in p2 if x not in child[s:e + 1]]
        pos = [(e + 1 + i) % n for i in range(n - (e - s + 1))]
        for idx, val in zip(pos, fill):
            child[idx] = val
        return child

    def _mutate(self, route):
        r = list(route)
        i, j = random.sample(range(len(r)), 2)
        r[i], r[j] = r[j], r[i]
        return r

    def run(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        pop = [self._random_route() for _ in range(self.pop_size)]
        history = []
        best_sol = None
        best_q = float("inf")

        for gen in range(self.max_gen):
            fits = [self._fitness(r) for r in pop]
            best_idx = int(np.argmin(fits))
            if fits[best_idx] < best_q:
                best_q = fits[best_idx]
                best_sol = list(pop[best_idx])
            history.append(best_q)

            # Elitism
            sorted_idx = np.argsort(fits)
            elites = [list(pop[i]) for i in sorted_idx[:self.elite_size]]

            new_pop = elites[:]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                child = self._ox(p1, p2) if random.random() < self.cx_rate else list(p1)
                if random.random() < self.mut_rate:
                    child = self._mutate(child)
                new_pop.append(child)
            pop = new_pop

        return _run_result(best_sol, best_q, history, time.perf_counter() - t0)


# ─── Simulated Annealing ──────────────────────────────────────────────────────

class SimulatedAnnealing:
    """
    SA with adaptive cooling schedule and 2-opt neighbourhood.
    """

    def __init__(
        self,
        tsp: TSPInstance,
        max_iterations: int = 50000,
        T_start: float = 100.0,
        T_end: float = 0.01,
        cooling: str = "geometric",  # 'geometric' | 'linear'
        seed: int = 42,
    ):
        self.tsp = tsp
        self.max_iter = max_iterations
        self.T_start = T_start
        self.T_end = T_end
        self.cooling = cooling
        random.seed(seed)

    def _neighbour(self, route):
        r = list(route)
        n = len(r)
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        r[i:j + 1] = r[i:j + 1][::-1]
        return r

    def run(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        current = list(range(self.tsp.n))
        random.shuffle(current)
        current_q = self.tsp.route_distance(current)
        best_sol = list(current)
        best_q = current_q

        history = []
        alpha = (self.T_end / self.T_start) ** (1.0 / self.max_iter)

        T = self.T_start
        for i in range(self.max_iter):
            if self.cooling == "geometric":
                T *= alpha
            else:
                T = self.T_start - (self.T_start - self.T_end) * i / self.max_iter

            T = max(T, 1e-10)
            neighbour = self._neighbour(current)
            n_q = self.tsp.route_distance(neighbour)
            delta = n_q - current_q
            if delta < 0 or random.random() < math.exp(-delta / T):
                current = neighbour
                current_q = n_q
            if current_q < best_q:
                best_q = current_q
                best_sol = list(current)

            # Record every 100 iters to match iteration-based history
            if i % 100 == 0:
                history.append(best_q)

        return _run_result(best_sol, best_q, history, time.perf_counter() - t0)


# ─── Particle Swarm Optimisation (discrete adaptation) ───────────────────────

class DiscreteParticleSwarm:
    """
    Discrete PSO for TSP.
    Velocity is interpreted as a sequence of swap operations.
    Position update: apply velocity swaps probabilistically.
    """

    def __init__(
        self,
        tsp: TSPInstance,
        swarm_size: int = 50,
        max_iterations: int = 500,
        inertia: float = 0.7,
        cognitive: float = 1.4,
        social: float = 1.4,
        seed: int = 42,
    ):
        self.tsp = tsp
        self.swarm_size = swarm_size
        self.max_iter = max_iterations
        self.w = inertia
        self.c1 = cognitive
        self.c2 = social
        random.seed(seed)
        np.random.seed(seed)

    def _random_route(self):
        r = list(range(self.tsp.n))
        random.shuffle(r)
        return r

    def _apply_swaps(self, route, swaps):
        r = list(route)
        for i, j in swaps:
            r[i], r[j] = r[j], r[i]
        return r

    def _route_to_swaps(self, r1, r2, probability=0.5):
        """Compute swap sequence to transform r1 into r2 stochastically."""
        n = len(r1)
        pos = {city: idx for idx, city in enumerate(r1)}
        swaps = []
        r = list(r1)
        for i in range(n):
            if r[i] != r2[i]:
                j = pos[r2[i]]
                if random.random() < probability:
                    swaps.append((i, j))
                    pos[r[i]] = j
                    pos[r[j]] = i
                    r[i], r[j] = r[j], r[i]
        return swaps

    def run(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        positions = [self._random_route() for _ in range(self.swarm_size)]
        personal_best = [list(p) for p in positions]
        personal_best_q = [self.tsp.route_distance(p) for p in personal_best]

        g_best_idx = int(np.argmin(personal_best_q))
        global_best = list(personal_best[g_best_idx])
        global_best_q = personal_best_q[g_best_idx]
        history = []

        for iteration in range(self.max_iter):
            for i in range(self.swarm_size):
                # Cognitive component: move toward personal best
                cog_swaps = self._route_to_swaps(
                    positions[i], personal_best[i], self.c1 * random.random()
                )
                # Social component: move toward global best
                soc_swaps = self._route_to_swaps(
                    positions[i], global_best, self.c2 * random.random()
                )
                # Apply inertia: random 2-opt if inertia kicks in
                new_pos = self._apply_swaps(positions[i], cog_swaps + soc_swaps)
                if random.random() < self.w:
                    a, b = sorted(random.sample(range(self.tsp.n), 2))
                    new_pos[a:b + 1] = new_pos[a:b + 1][::-1]

                positions[i] = new_pos
                q = self.tsp.route_distance(new_pos)

                if q < personal_best_q[i]:
                    personal_best[i] = list(new_pos)
                    personal_best_q[i] = q

                if q < global_best_q:
                    global_best = list(new_pos)
                    global_best_q = q

            history.append(global_best_q)

        return _run_result(global_best, global_best_q, history, time.perf_counter() - t0)


# ─── Ant Colony Optimisation ──────────────────────────────────────────────────

class AntColonyOptimisation:
    """
    AS (Ant System) variant of ACO for TSP.
    Pheromone-guided probabilistic construction + evaporation.
    """

    def __init__(
        self,
        tsp: TSPInstance,
        n_ants: int = 30,
        max_iterations: int = 500,
        alpha: float = 1.0,    # pheromone importance
        beta: float = 3.0,     # heuristic importance
        rho: float = 0.1,      # evaporation rate
        Q: float = 100.0,      # pheromone deposit constant
        seed: int = 42,
    ):
        self.tsp = tsp
        self.n_ants = n_ants
        self.max_iter = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        random.seed(seed)
        np.random.seed(seed)

        n = tsp.n
        # Initialise pheromone uniformly
        self.pheromone = np.ones((n, n)) * 0.1
        # Heuristic: inverse distance (avoid division by zero on diagonal)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.heuristic = np.where(
                tsp.dist_matrix > 0, 1.0 / tsp.dist_matrix, 0.0
            )

    def _build_route(self, start: int) -> List[int]:
        n = self.tsp.n
        visited = [False] * n
        route = [start]
        visited[start] = True
        for _ in range(n - 1):
            current = route[-1]
            probs = np.zeros(n)
            for j in range(n):
                if not visited[j]:
                    probs[j] = (self.pheromone[current, j] ** self.alpha) * \
                               (self.heuristic[current, j] ** self.beta)
            total = probs.sum()
            if total == 0:
                unvisited = [j for j in range(n) if not visited[j]]
                next_city = random.choice(unvisited)
            else:
                probs /= total
                next_city = int(np.random.choice(n, p=probs))
            route.append(next_city)
            visited[next_city] = True
        return route

    def _update_pheromone(self, routes, qualities):
        self.pheromone *= (1 - self.rho)
        for route, q in zip(routes, qualities):
            deposit = self.Q / q
            for i in range(len(route)):
                a, b = route[i], route[(i + 1) % len(route)]
                self.pheromone[a, b] += deposit
                self.pheromone[b, a] += deposit
        np.clip(self.pheromone, 1e-6, None, out=self.pheromone)

    def run(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        best_sol = None
        best_q = float("inf")
        history = []

        for iteration in range(self.max_iter):
            routes = [self._build_route(random.randint(0, self.tsp.n - 1))
                      for _ in range(self.n_ants)]
            qualities = [self.tsp.route_distance(r) for r in routes]

            best_idx = int(np.argmin(qualities))
            if qualities[best_idx] < best_q:
                best_q = qualities[best_idx]
                best_sol = list(routes[best_idx])

            self._update_pheromone(routes, qualities)
            history.append(best_q)

        return _run_result(best_sol, best_q, history, time.perf_counter() - t0)


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_all_baselines(
    tsp: TSPInstance,
    max_iterations: int = 500,
    population_size: int = 50,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function: run all four baselines on a TSP instance.
    Returns dict: algo_name → result_dict.
    """
    print("  Running GA...")
    ga = GeneticAlgorithm(tsp, population_size=population_size,
                          max_generations=max_iterations, seed=seed)
    ga_res = ga.run()

    print("  Running SA...")
    sa = SimulatedAnnealing(tsp, max_iterations=max_iterations * 100, seed=seed)
    sa_res = sa.run()

    print("  Running PSO...")
    pso = DiscreteParticleSwarm(tsp, swarm_size=population_size,
                                max_iterations=max_iterations, seed=seed)
    pso_res = pso.run()

    print("  Running ACO...")
    aco = AntColonyOptimisation(tsp, n_ants=population_size // 2,
                                max_iterations=max_iterations, seed=seed)
    aco_res = aco.run()

    return {
        "GA": ga_res,
        "SA": sa_res,
        "PSO": pso_res,
        "ACO": aco_res,
    }
