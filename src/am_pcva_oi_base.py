from __future__ import annotations

import csv
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

try:
    import tsplib95  # type: ignore
except Exception:  # pragma: no cover
    tsplib95 = None

Tour = List[int]


@dataclass
class Individual:
    tour: Tour
    cost: float
    generations_without_improvement: int = 0
    age: int = 0


@dataclass
class DecisionRecord:
    generation: int
    individual_rank: int
    individual_cost: float
    best_cost: float
    worst_cost: float
    mean_cost: float
    std_cost: float
    normalized_cost: float
    relative_gap_to_best: float
    age: int
    stagnation: int
    unique_edges_ratio: float
    mean_edge_cost: float
    max_edge_cost: float
    local_search_applied: int
    improved: int
    delta_cost: float
    local_search_time_ms: float


@dataclass
class AMPCVAOIConfig:
    population_size: int = 10
    generations: int = 300
    crossover_rate: float = 1.0
    mutation_rate: float = 0.08
    tournament_size: int = 3
    elite_size: int = 1
    seed: int = 42
    nearest_neighbor_seeds: int = 3
    local_search_mode: str = "2opt"  # use "none" if you want pure GA steps only
    local_search_on_children: bool = True
    two_opt_first_improvement: bool = True
    two_opt_max_passes: int = 20
    stagnation_limit: int = 60
    time_limit_seconds: Optional[float] = None


class DistanceMatrix:
    def __init__(self, matrix: Sequence[Sequence[float]]):
        if not matrix or len(matrix) != len(matrix[0]):
            raise ValueError("Distance matrix must be square and non-empty.")
        self.matrix = [list(map(float, row)) for row in matrix]
        self.n = len(self.matrix)

    def d(self, i: int, j: int) -> float:
        return self.matrix[i][j]

    def tour_cost(self, tour: Sequence[int]) -> float:
        total = 0.0
        n = len(tour)
        for idx in range(n):
            total += self.matrix[tour[idx]][tour[(idx + 1) % n]]
        return total

    @classmethod
    def from_tsplib(cls, filepath: str | Path) -> "DistanceMatrix":
        if tsplib95 is None:
            raise ImportError("tsplib95 is not installed. Install it with: pip install tsplib95")
        problem = tsplib95.load(str(filepath))
        n = problem.dimension
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 0.0
                else:
                    matrix[i][j] = float(problem.get_weight(i + 1, j + 1))
        return cls(matrix)


class LocalSearchPolicy:
    """Policy hook prepared for XGBoost/LightGBM.

    Return True when local search should be applied to a child.
    """

    def should_apply(self, features: dict) -> bool:
        raise NotImplementedError


class ExploratoryLocalSearchPolicy(LocalSearchPolicy):

    def __init__(self, rng: random.Random):
        self.rng = rng

    def should_apply(self, features: dict) -> bool:

        rank = int(features["individual_rank"])
        gap = float(features["relative_gap_to_best"])
        stagnation = int(features["stagnation"])

        prob = 0.30

        # soluções com gap moderado costumam melhorar
        if 0.01 < gap < 0.20:
            prob += 0.25

        # indivíduos estagnados também podem melhorar
        if stagnation >= 2:
            prob += 0.15

        prob = max(0.10, min(0.90, prob))

        return self.rng.random() < prob


class ThresholdPolicy(LocalSearchPolicy):
    def __init__(self, threshold: float = 0.5, score_fn: Optional[Callable[[dict], float]] = None):
        self.threshold = threshold
        self.score_fn = score_fn or (lambda _: 1.0)

    def should_apply(self, features: dict) -> bool:
        return float(self.score_fn(features)) >= self.threshold


class AMPCVAOI:
    """Best-effort reconstruction scaffold inspired by AM_PCVA-OI.

    Faithful parts:
    - permutation representation
    - OX1 crossover
    - ISM mutation
    - memetic GA structure
    - population size configurable (10 by default)

    Important note:
    The thesis reports LK-based local refinement. A full Lin-Kernighan implementation is
    non-trivial and not fully specified in the dissertation text alone. For a reliable base
    implementation, this scaffold uses 2-opt by default and leaves a clean hook for replacing
    it with a true LK implementation later.
    """

    def __init__(
        self,
        dist: DistanceMatrix,
        config: Optional[AMPCVAOIConfig] = None,
        policy: Optional[LocalSearchPolicy] = None,
        collect_decisions: bool = False,
    ):
        self.dist = dist
        self.cfg = config or AMPCVAOIConfig()
        self.rng = random.Random(self.cfg.seed)
        self.policy = policy or ExploratoryLocalSearchPolicy(self.rng)
        self.collect_decisions = collect_decisions
        self.decision_records: List[DecisionRecord] = []

    # ------------------------- public API -------------------------
    def run(self) -> Individual:
        population = self._initialize_population()
        population.sort(key=lambda ind: ind.cost)
        best = self._clone_individual(population[0])

        start = time.perf_counter()
        stagnant_generations = 0

        for generation in range(self.cfg.generations):
            if self.cfg.time_limit_seconds is not None:
                if (time.perf_counter() - start) >= self.cfg.time_limit_seconds:
                    break

            next_population: List[Individual] = []

            elites = [self._clone_individual(ind) for ind in population[: self.cfg.elite_size]]
            for elite in elites:
                elite.age += 1
            next_population.extend(elites)

            while len(next_population) < self.cfg.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                if self.rng.random() <= self.cfg.crossover_rate:
                    child_tour_1, child_tour_2 = self._ox1(parent1.tour, parent2.tour)
                else:
                    child_tour_1, child_tour_2 = list(parent1.tour), list(parent2.tour)

                child_tour_1 = self._ism_mutation(child_tour_1)
                child_tour_2 = self._ism_mutation(child_tour_2)

                child_age = max(parent1.age, parent2.age) + 1
                child_stagnation = min(
                    parent1.generations_without_improvement,
                    parent2.generations_without_improvement
                ) + 1

                child1 = Individual(
                    tour=child_tour_1,
                    cost=self.dist.tour_cost(child_tour_1),
                    generations_without_improvement=child_stagnation,
                    age=child_age,
                )

                child2 = Individual(
                    tour=child_tour_2,
                    cost=self.dist.tour_cost(child_tour_2),
                    generations_without_improvement=child_stagnation,
                    age=child_age,
                )

                if self.cfg.local_search_on_children:
                    child1 = self._maybe_apply_local_search(child1, population, generation)
                    child2 = self._maybe_apply_local_search(child2, population, generation)

                next_population.append(child1)
                if len(next_population) < self.cfg.population_size:
                    next_population.append(child2)

            population = sorted(next_population, key=lambda ind: ind.cost)
            current_best = population[0]

            if current_best.cost + 1e-12 < best.cost:
                best = self._clone_individual(current_best)
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            if stagnant_generations >= self.cfg.stagnation_limit:
                break

        return best

    def export_decision_dataset(self, filepath: str | Path) -> None:
        if not self.decision_records:
            raise ValueError("No decision records were collected. Run with collect_decisions=True.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(DecisionRecord.__annotations__.keys()))
            for rec in self.decision_records:
                writer.writerow([getattr(rec, k) for k in DecisionRecord.__annotations__.keys()])

    # ------------------------- initialization -------------------------
    def _initialize_population(self) -> List[Individual]:
        n = self.dist.n
        seeds = min(self.cfg.nearest_neighbor_seeds, self.cfg.population_size)
        population: List[Individual] = []

        used_signatures = set()

        for start in range(seeds):
            tour = self._nearest_neighbor_tour(start)
            signature = tuple(tour)
            if signature not in used_signatures:
                used_signatures.add(signature)
                population.append(Individual(tour=tour, cost=self.dist.tour_cost(tour)))

        while len(population) < self.cfg.population_size:
            tour = list(range(n))
            self.rng.shuffle(tour)
            signature = tuple(tour)
            if signature in used_signatures:
                continue
            used_signatures.add(signature)
            population.append(Individual(tour=tour, cost=self.dist.tour_cost(tour)))

            if self.cfg.local_search_mode != "none" and not self.collect_decisions:
                population = [self._apply_local_search(ind) for ind in population]

        return sorted(population, key=lambda ind: ind.cost)

    def _nearest_neighbor_tour(self, start: int) -> Tour:
        n = self.dist.n
        unvisited = set(range(n))
        current = start % n
        tour = [current]
        unvisited.remove(current)

        while unvisited:
            nxt = min(unvisited, key=lambda j: self.dist.d(current, j))
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return tour

    # ------------------------- GA operators -------------------------
    def _tournament_selection(self, population: Sequence[Individual]) -> Individual:
        contestants = self.rng.sample(list(population), k=min(self.cfg.tournament_size, len(population)))
        return min(contestants, key=lambda ind: ind.cost)

    def _ox1(self, p1: Sequence[int], p2: Sequence[int]) -> Tuple[Tour, Tour]:
        n = len(p1)
        a, b = sorted(self.rng.sample(range(n), 2))

        def build_child(parent_a: Sequence[int], parent_b: Sequence[int]) -> Tour:
            child = [-1] * n
            child[a : b + 1] = parent_a[a : b + 1]
            taken = set(child[a : b + 1])
            fill_values = [gene for gene in parent_b if gene not in taken]
            fill_idx = 0
            for i in list(range(0, a)) + list(range(b + 1, n)):
                child[i] = fill_values[fill_idx]
                fill_idx += 1
            return child

        return build_child(p1, p2), build_child(p2, p1)

    def _ism_mutation(self, tour: Tour) -> Tour:
        if self.rng.random() > self.cfg.mutation_rate:
            return tour
        n = len(tour)
        i, j = self.rng.sample(range(n), 2)
        mutated = list(tour)
        node = mutated.pop(i)
        mutated.insert(j, node)
        return mutated

    # ------------------------- local search -------------------------
    def _maybe_apply_local_search(
        self,
        individual: Individual,
        population: Sequence[Individual],
        generation: int,
    ) -> Individual:
        features = self._extract_features(individual, population, generation)

        rank = int(features["individual_rank"])
        gap = float(features["relative_gap_to_best"])
        stagnation = int(features["stagnation"])

        # evita ruído inicial muito instável
        if generation < 2:
            return individual

        # -------------------------
        # FASE 1: FORÇAR DIVERSIDADE
        # -------------------------
        force_apply = False

        # 1. indivíduos médios (melhor zona de aprendizado)
        if 2 <= rank <= 8:
            force_apply = True

        # 2. indivíduos piores (alta chance de melhora)
        elif gap > 0.10:
            force_apply = True

        # 3. indivíduos estagnados
        elif stagnation >= 2:
            force_apply = True

        # 4. exploração aleatória mais agressiva
        elif self.rng.random() < 0.35:
            force_apply = True

        # evita excesso de elite
        if rank == 1 and self.rng.random() > 0.30:
            force_apply = False

        if not force_apply:
            if self.collect_decisions:
                self._record_decision(features, 0, 0, 0.0, 0.0)
            return individual

        # -------------------------
        # FASE 2: POLICY (suavizada)
        # -------------------------
        # deixa a policy menos restritiva para gerar mais positivos
        if self.policy:
            if self.rng.random() > 0.85:  # bypass parcial da policy
                should_apply = True
            else:
                should_apply = self.policy.should_apply(features)
        else:
            should_apply = True

        if not should_apply:
            if self.collect_decisions:
                self._record_decision(features, 0, 0, 0.0, 0.0)
            return individual

        # -------------------------
        # FASE 3: EXECUÇÃO REAL
        # -------------------------
        before = individual.cost
        t0 = time.perf_counter()
        improved_ind = self._apply_local_search(individual)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        delta = before - improved_ind.cost
        improved = 1 if delta > 1e-12 else 0

        if self.collect_decisions:
            self._record_decision(features, 1, improved, delta, elapsed_ms)

        if improved:
            improved_ind.generations_without_improvement = 0
        else:
            improved_ind.generations_without_improvement = stagnation + 1

        return improved_ind

    def _apply_local_search(self, individual: Individual) -> Individual:
        mode = self.cfg.local_search_mode.lower()
        if mode == "none":
            return self._clone_individual(individual)
        if mode == "2opt":
            return self._two_opt(individual)
        raise ValueError(f"Unsupported local_search_mode: {self.cfg.local_search_mode}")

    def _two_opt(self, individual: Individual) -> Individual:
        best_tour = list(individual.tour)
        best_cost = individual.cost
        n = len(best_tour)
        passes = 0

        improved = True
        while improved and passes < self.cfg.two_opt_max_passes:
            improved = False
            passes += 1
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    if j - i == 1:
                        continue
                    candidate = best_tour[:i] + list(reversed(best_tour[i:j])) + best_tour[j:]
                    candidate_cost = self.dist.tour_cost(candidate)
                    if candidate_cost + 1e-12 < best_cost:
                        best_tour, best_cost = candidate, candidate_cost
                        improved = True
                        if self.cfg.two_opt_first_improvement:
                            break
                if improved and self.cfg.two_opt_first_improvement:
                    break
        return Individual(
            tour=best_tour,
            cost=best_cost,
            generations_without_improvement=individual.generations_without_improvement,
            age=individual.age,
        )

    # ------------------------- features / logs -------------------------
    def _extract_features(
        self,
        individual: Individual,
        population: Sequence[Individual],
        generation: int,
    ) -> dict:
        costs = [ind.cost for ind in population]
        best_cost = min(costs)
        worst_cost = max(costs)
        mean_cost = mean(costs)
        std_cost = pstdev(costs) if len(costs) > 1 else 0.0
        rank = 1 + sum(1 for c in costs if c < individual.cost)
        unique_edges_ratio = self._unique_edges_ratio(individual.tour, [ind.tour for ind in population])
        edge_costs = [self.dist.d(individual.tour[k], individual.tour[(k + 1) % len(individual.tour)]) for k in range(len(individual.tour))]
        denom = (worst_cost - best_cost) if worst_cost > best_cost else 1.0

        return {
            "generation": generation,
            "individual_rank": rank,
            "individual_cost": individual.cost,
            "best_cost": best_cost,
            "worst_cost": worst_cost,
            "mean_cost": mean_cost,
            "std_cost": std_cost,
            "normalized_cost": (individual.cost - best_cost) / denom,
            "relative_gap_to_best": (individual.cost - best_cost) / max(best_cost, 1e-12),
            "age": individual.age,
            "stagnation": individual.generations_without_improvement,
            "unique_edges_ratio": unique_edges_ratio,
            "mean_edge_cost": mean(edge_costs),
            "max_edge_cost": max(edge_costs),
        }

    def _record_decision(self, features: dict, applied: int, improved: int, delta_cost: float, elapsed_ms: float) -> None:
        self.decision_records.append(
            DecisionRecord(
                generation=int(features["generation"]),
                individual_rank=int(features["individual_rank"]),
                individual_cost=float(features["individual_cost"]),
                best_cost=float(features["best_cost"]),
                worst_cost=float(features["worst_cost"]),
                mean_cost=float(features["mean_cost"]),
                std_cost=float(features["std_cost"]),
                normalized_cost=float(features["normalized_cost"]),
                relative_gap_to_best=float(features["relative_gap_to_best"]),
                age=int(features["age"]),
                stagnation=int(features["stagnation"]),
                unique_edges_ratio=float(features["unique_edges_ratio"]),
                mean_edge_cost=float(features["mean_edge_cost"]),
                max_edge_cost=float(features["max_edge_cost"]),
                local_search_applied=int(applied),
                improved=int(improved),
                delta_cost=float(delta_cost),
                local_search_time_ms=float(elapsed_ms),
            )
        )

    def _unique_edges_ratio(self, tour: Sequence[int], population_tours: Sequence[Sequence[int]]) -> float:
        edges = self._tour_edges(tour)
        edge_counter = {}
        for other in population_tours:
            for edge in self._tour_edges(other):
                edge_counter[edge] = edge_counter.get(edge, 0) + 1
        unique_edges = sum(1 for edge in edges if edge_counter.get(edge, 0) == 1)
        return unique_edges / max(len(edges), 1)

    @staticmethod
    def _tour_edges(tour: Sequence[int]) -> List[Tuple[int, int]]:
        return [(tour[i], tour[(i + 1) % len(tour)]) for i in range(len(tour))]

    @staticmethod
    def _clone_individual(ind: Individual) -> Individual:
        return Individual(
            tour=list(ind.tour),
            cost=float(ind.cost),
            generations_without_improvement=int(ind.generations_without_improvement),
            age=int(ind.age),
        )


def random_euclidean_instance(n: int, seed: int = 42) -> DistanceMatrix:
    rng = random.Random(seed)
    points = [(rng.random(), rng.random()) for _ in range(n)]
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            matrix[i][j] = math.hypot(xi - xj, yi - yj)
    return DistanceMatrix(matrix)


def main() -> None:
    dist = random_euclidean_instance(n=30, seed=7)
    config = AMPCVAOIConfig(
        population_size=10,
        generations=200,
        mutation_rate=0.08,
        local_search_mode="2opt",
        seed=7,
    )
    solver = AMPCVAOI(dist, config=config, collect_decisions=True)
    best = solver.run()
    print("[DEBUG] Melhor custo | Best cost:", round(best.cost, 6))
    print("[DEBUG] Melhor tour | Best tour:", best.tour)
    solver.export_decision_dataset("data/decision_dataset.csv")
    print("[INFO] Dataset de decisão salvo em decision_dataset.csv | Decision dataset saved to decision_dataset.csv")


if __name__ == "__main__":
    main()