from __future__ import annotations

import argparse
import os
from pathlib import Path

from am_pcva_oi_base import (
    AMPCVAOI,
    AMPCVAOIConfig,
    ClassicalGroverSearchBackend,
    QiskitGroverSearchBackend,
    random_euclidean_instance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKEND_KIND = os.getenv("AM_PCVA_GROVER_BACKEND", "classical")


def build_backend(kind: str):
    normalized = kind.lower()
    if normalized == "classical":
        return ClassicalGroverSearchBackend()
    if normalized == "qiskit_statevector":
        return QiskitGroverSearchBackend()
    raise ValueError(f"Unsupported Grover backend: {kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa a variante hibrida AM-PCVA-OI + Grover.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND_KIND,
        choices=["classical", "qiskit_statevector"],
        help="Backend de busca Grover.",
    )
    parser.add_argument("--instance-size", type=int, default=50, help="Numero de cidades da instancia aleatoria.")
    parser.add_argument("--instance-seed", type=int, default=7, help="Seed da instancia aleatoria.")
    parser.add_argument("--solver-seed", type=int, default=7, help="Seed do solver.")
    parser.add_argument("--population-size", type=int, default=10, help="Tamanho da populacao.")
    parser.add_argument("--generations", type=int, default=200, help="Numero maximo de geracoes.")
    parser.add_argument("--mutation-rate", type=float, default=0.08, help="Taxa de mutacao.")
    parser.add_argument("--pool-size", type=int, default=64, help="Tamanho do pool de movimentos 2-opt.")
    parser.add_argument("--shots", type=int, default=1024, help="Numero de shots usado pelo backend Grover.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist = random_euclidean_instance(n=args.instance_size, seed=args.instance_seed)
    backend = build_backend(args.backend)

    config = AMPCVAOIConfig(
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        local_search_mode="grover_2opt",
        seed=args.solver_seed,
        grover_candidate_pool_size=args.pool_size,
        grover_shots=args.shots,
    )

    solver = AMPCVAOI(
        dist=dist,
        config=config,
        policy=None,
        grover_backend=backend,
        collect_decisions=False,
    )

    best = solver.run()
    grover_stats = solver.get_grover_stats()

    print("=" * 70)
    print("[RESULTADO FINAL - AM_PCVA_OI + GROVER]")
    print(f"Instancia: n={args.instance_size} | instance_seed={args.instance_seed} | solver_seed={args.solver_seed}")
    print(f"Backend Grover: {grover_stats['backend']}")
    print(f"Melhor custo: {best.cost:.6f}")
    print(f"Melhor tour: {best.tour}")
    print(f"Chamadas Grover: {grover_stats['calls']}")
    print(f"Sucessos Grover: {grover_stats['successes']}")
    print(f"Pool medio de candidatos: {grover_stats['mean_candidate_pool_size']:.2f}")
    print(f"Tempo medio backend (ms): {grover_stats['mean_backend_time_ms']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
