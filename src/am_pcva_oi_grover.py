from __future__ import annotations

from pathlib import Path

from am_pcva_oi_base import (
    AMPCVAOI,
    AMPCVAOIConfig,
    ClassicalGroverSearchBackend,
    QiskitGroverSearchBackend,
    random_euclidean_instance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_KIND = "classical"


def build_backend(kind: str):
    normalized = kind.lower()
    if normalized == "classical":
        return ClassicalGroverSearchBackend()
    if normalized == "qiskit_statevector":
        return QiskitGroverSearchBackend()
    raise ValueError(f"Unsupported Grover backend: {kind}")


def main() -> None:
    dist = random_euclidean_instance(n=50, seed=7)
    backend = build_backend(BACKEND_KIND)

    config = AMPCVAOIConfig(
        population_size=10,
        generations=200,
        mutation_rate=0.08,
        local_search_mode="grover_2opt",
        seed=7,
        grover_candidate_pool_size=64,
        grover_shots=1024,
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
