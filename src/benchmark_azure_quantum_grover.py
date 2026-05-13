from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from statistics import mean

import pandas as pd

from am_pcva_oi_base import AMPCVAOI, AMPCVAOIConfig, random_euclidean_instance
from azure_quantum_grover_runner import run_azure_grover_job


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAILS_CSV = OUTPUT_DIR / "benchmark_azure_quantum_grover_detailed.csv"
SUMMARY_CSV = OUTPUT_DIR / "benchmark_azure_quantum_grover_summary.csv"


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for chunk in value.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            result.extend(list(range(start, end + step, step)))
        else:
            result.append(int(token))
    return result


def build_solver(dist, solver_seed: int, pool_size: int, shots: int) -> AMPCVAOI:
    config = AMPCVAOIConfig(
        population_size=10,
        generations=200,
        mutation_rate=0.08,
        local_search_mode="grover_2opt",
        seed=solver_seed,
        grover_candidate_pool_size=pool_size,
        grover_shots=shots,
    )
    return AMPCVAOI(dist=dist, config=config, policy=None, collect_decisions=False)


def build_cases(solver: AMPCVAOI, cases_per_run: int) -> list[dict]:
    original_mode = solver.cfg.local_search_mode
    solver.cfg.local_search_mode = "none"
    try:
        population = sorted(solver._initialize_population(), key=lambda ind: ind.cost)
    finally:
        solver.cfg.local_search_mode = original_mode
    cases: list[dict] = []

    for individual_rank, individual in enumerate(population[:cases_per_run], start=1):
        moves = solver._ranked_two_opt_moves(individual.tour)
        if not moves:
            continue

        candidate_moves = moves[: solver.cfg.grover_candidate_pool_size]
        improvements = [move.improvement for move in candidate_moves]
        good_indices = [idx for idx, improvement in enumerate(improvements) if improvement > 1e-12]
        if not good_indices:
            continue

        positive_improvements = [improvement for improvement in improvements if improvement > 1e-12]
        cases.append(
            {
                "individual_rank": individual_rank,
                "individual_cost": float(individual.cost),
                "num_items": len(candidate_moves),
                "num_qubits": max(1, math.ceil(math.log2(max(len(candidate_moves), 2)))),
                "num_good_indices": len(good_indices),
                "good_indices": good_indices,
                "best_improvement": float(improvements[0]),
                "mean_positive_improvement": float(mean(positive_improvements)),
            }
        )

    return cases


def run_case(case: dict, args: argparse.Namespace, instance_size: int, instance_seed: int, solver_seed: int) -> dict:
    t0 = time.perf_counter()
    run_info = run_azure_grover_job(
        resource_id=args.resource_id,
        target=args.target,
        tenant_id=args.tenant_id,
        auth_debug=args.auth_debug,
        num_items=case["num_items"],
        good_indices=case["good_indices"],
        shots=args.shots,
        iterations=args.iterations,
        job_name=(
            f"Azure Grover Benchmark n={instance_size} "
            f"inst={instance_seed} solver={solver_seed} rank={case['individual_rank']}"
        ),
    )
    elapsed = time.perf_counter() - t0

    return {
        "approach": "am_pcva_oi_grover_azure_quantum",
        "instance_size": instance_size,
        "instance_seed": instance_seed,
        "solver_seed": solver_seed,
        "individual_rank": case["individual_rank"],
        "individual_cost": case["individual_cost"],
        "num_items": case["num_items"],
        "num_qubits": case["num_qubits"],
        "num_good_indices": case["num_good_indices"],
        "good_indices": ",".join(str(idx) for idx in case["good_indices"]),
        "best_improvement": case["best_improvement"],
        "mean_positive_improvement": case["mean_positive_improvement"],
        "job_id": run_info["job_id"],
        "target": run_info["target"],
        "shots": run_info["shots"],
        "iterations": run_info["iterations"],
        "top_state": run_info["top_state"],
        "top_index": run_info["top_index"],
        "top_count": run_info["top_count"],
        "valid_item": int(run_info["valid_item"]),
        "marked_solution": int(run_info["marked_solution"]),
        "best_move_hit": int(run_info["top_index"] == 0),
        "counts": run_info["formatted_counts"],
        "runtime_seconds": float(elapsed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multi-instancia para o job isolado de Grover no Azure Quantum."
    )
    parser.add_argument("--resource-id", required=True, help="Resource ID do workspace Azure Quantum.")
    parser.add_argument("--target", required=True, help="Nome do target Azure Quantum.")
    parser.add_argument("--tenant-id", default="", help="Tenant ID usado para autenticacao no Azure.")
    parser.add_argument(
        "--auth-debug",
        action="store_true",
        help="Imprime qual credencial Azure foi selecionada durante os jobs.",
    )
    parser.add_argument("--instance-sizes", default="30,40", help="Lista ou intervalo de tamanhos, ex.: 30,40 ou 30-60.")
    parser.add_argument("--instance-seeds", default="1-2", help="Lista ou intervalo de seeds das instancias.")
    parser.add_argument("--solver-seeds", default="1-2", help="Lista ou intervalo de seeds do solver.")
    parser.add_argument(
        "--cases-per-run",
        type=int,
        default=2,
        help="Quantidade de individuos avaliados por combinacao de instancia e seed.",
    )
    parser.add_argument("--pool-size", type=int, default=64, help="Tamanho do pool de candidatos 2-opt.")
    parser.add_argument("--shots", type=int, default=1024, help="Numero de shots por job quantico.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Numero fixo de iteracoes de Grover. Se omitido, usa a estimativa teorica.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance_sizes = parse_int_list(args.instance_sizes)
    instance_seeds = parse_int_list(args.instance_seeds)
    solver_seeds = parse_int_list(args.solver_seeds)

    results: list[dict] = []
    pending_cases: list[tuple[int, int, int, dict]] = []

    for instance_size in instance_sizes:
        for instance_seed in instance_seeds:
            dist = random_euclidean_instance(n=instance_size, seed=instance_seed)
            for solver_seed in solver_seeds:
                solver = build_solver(dist, solver_seed, args.pool_size, args.shots)
                for case in build_cases(solver, args.cases_per_run):
                    pending_cases.append((instance_size, instance_seed, solver_seed, case))

    total_cases = len(pending_cases)
    if total_cases == 0:
        raise RuntimeError("Nenhum caso com movimentos melhorantes foi encontrado para o benchmark.")

    print("=" * 80)
    print("[START] Benchmark Azure Quantum Grover")
    print(f"[INFO] target={args.target}")
    print(f"[INFO] instance_sizes={instance_sizes}")
    print(f"[INFO] instance_seeds={instance_seeds}")
    print(f"[INFO] solver_seeds={solver_seeds}")
    print(f"[INFO] cases_per_run={args.cases_per_run}")
    print(f"[INFO] total_jobs={total_cases}")
    print("=" * 80)

    for current, (instance_size, instance_seed, solver_seed, case) in enumerate(pending_cases, start=1):
        print(
            f"[{current}/{total_cases}] "
            f"n={instance_size} | inst_seed={instance_seed} | solver_seed={solver_seed} | "
            f"rank={case['individual_rank']} | items={case['num_items']} | good={case['num_good_indices']}"
        )
        results.append(run_case(case, args, instance_size, instance_seed, solver_seed))

    df = pd.DataFrame(results)
    df.to_csv(DETAILS_CSV, index=False)

    summary = (
        df.groupby("approach")
        .agg(
            runs=("job_id", "count"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_num_items=("num_items", "mean"),
            mean_num_good_indices=("num_good_indices", "mean"),
            valid_item_rate=("valid_item", "mean"),
            marked_solution_rate=("marked_solution", "mean"),
            best_move_hit_rate=("best_move_hit", "mean"),
        )
        .reset_index()
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
