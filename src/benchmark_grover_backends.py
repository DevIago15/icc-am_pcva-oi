from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from am_pcva_oi_base import (
    AMPCVAOI,
    AMPCVAOIConfig,
    ClassicalGroverSearchBackend,
    QiskitGroverSearchBackend,
    random_euclidean_instance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAILS_CSV = OUTPUT_DIR / "benchmark_grover_detailed.csv"
SUMMARY_CSV = OUTPUT_DIR / "benchmark_grover_summary.csv"

INSTANCE_SIZES = [30, 40, 50, 60]
INSTANCE_SEEDS = range(1, 11)
SOLVER_SEEDS = range(1, 6)


def qiskit_is_available() -> bool:
    try:
        import qiskit  # noqa: F401
        import qiskit_algorithms  # noqa: F401
    except Exception:
        return False
    return True


def build_base_solver(dist, seed):
    config = AMPCVAOIConfig(
        population_size=10,
        generations=200,
        mutation_rate=0.08,
        local_search_mode="2opt",
        seed=seed,
    )
    return AMPCVAOI(dist=dist, config=config, policy=None, collect_decisions=False)


def build_grover_solver(dist, seed, backend):
    config = AMPCVAOIConfig(
        population_size=10,
        generations=200,
        mutation_rate=0.08,
        local_search_mode="grover_2opt",
        seed=seed,
        grover_candidate_pool_size=64,
        grover_shots=1024,
    )
    return AMPCVAOI(
        dist=dist,
        config=config,
        policy=None,
        grover_backend=backend,
        collect_decisions=False,
    )


def run_solver(label, solver, instance_size, instance_seed, solver_seed):
    t0 = time.perf_counter()
    best = solver.run()
    elapsed = time.perf_counter() - t0
    grover_stats = solver.get_grover_stats()

    return {
        "approach": label,
        "instance_size": instance_size,
        "instance_seed": instance_seed,
        "solver_seed": solver_seed,
        "best_cost": float(best.cost),
        "runtime_seconds": float(elapsed),
        "grover_backend": grover_stats["backend"],
        "grover_calls": int(grover_stats["calls"]),
        "grover_successes": int(grover_stats["successes"]),
        "grover_mean_candidate_pool_size": float(grover_stats["mean_candidate_pool_size"]),
        "grover_total_backend_time_ms": float(grover_stats["total_backend_time_ms"]),
        "grover_mean_backend_time_ms": float(grover_stats["mean_backend_time_ms"]),
        "grover_total_improvement": float(grover_stats["total_improvement"]),
    }


def discover_approaches():
    approaches = [
        {
            "label": "am_pcva_oi_base_2opt",
            "builder": build_base_solver,
        },
        {
            "label": "am_pcva_oi_grover_classical",
            "builder": lambda dist, seed: build_grover_solver(dist, seed, ClassicalGroverSearchBackend()),
        },
    ]

    if qiskit_is_available():
        approaches.append(
            {
                "label": "am_pcva_oi_grover_qiskit_statevector",
                "builder": lambda dist, seed: build_grover_solver(dist, seed, QiskitGroverSearchBackend()),
            }
        )
    else:
        print("[SKIP] Backend qiskit_statevector indisponivel no ambiente atual")

    return approaches


def main() -> None:
    results = []
    approaches = discover_approaches()
    total_runs = len(INSTANCE_SIZES) * len(INSTANCE_SEEDS) * len(SOLVER_SEEDS) * len(approaches)
    current = 0

    print("=" * 80)
    print("[START] Benchmark Base 2-opt vs hibrido com Grover")
    print(f"[INFO] Total de abordagens: {len(approaches)}")
    print(f"[INFO] Total de execucoes: {total_runs}")
    print("=" * 80)

    for n in INSTANCE_SIZES:
        for inst_seed in INSTANCE_SEEDS:
            dist = random_euclidean_instance(n=n, seed=inst_seed)
            for solver_seed in SOLVER_SEEDS:
                for approach in approaches:
                    current += 1
                    print(
                        f"[{current}/{total_runs}] {approach['label']} | "
                        f"n={n} | inst_seed={inst_seed} | solver_seed={solver_seed}"
                    )
                    solver = approach["builder"](dist, solver_seed)
                    results.append(run_solver(approach["label"], solver, n, inst_seed, solver_seed))

    df = pd.DataFrame(results)
    df.to_csv(DETAILS_CSV, index=False)

    summary = (
        df.groupby("approach")
        .agg(
            runs=("best_cost", "count"),
            mean_best_cost=("best_cost", "mean"),
            median_best_cost=("best_cost", "median"),
            std_best_cost=("best_cost", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_grover_calls=("grover_calls", "mean"),
            mean_grover_successes=("grover_successes", "mean"),
            mean_grover_candidate_pool=("grover_mean_candidate_pool_size", "mean"),
            mean_grover_backend_time_ms=("grover_total_backend_time_ms", "mean"),
        )
        .reset_index()
        .sort_values(["mean_best_cost", "mean_runtime_seconds"], ascending=[True, True])
    )

    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print(summary.to_string(index=False))
    print("=" * 80)
    print(f"[DONE] Detalhado salvo em: {DETAILS_CSV}")
    print(f"[DONE] Resumo salvo em:    {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
