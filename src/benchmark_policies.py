from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import pandas as pd

from am_pcva_oi_base import (
    AMPCVAOI as BaseAMPCVAOI,
    AMPCVAOIConfig as BaseConfig,
    random_euclidean_instance,
)
from am_pcva_oi_lightgbm import (
    AMPCVAOI as LGBMAMPCVAOI,
    AMPCVAOIConfig as LGBMConfig,
    LightGBMLocalSearchPolicy,
)
from am_pcva_oi_xgboost import (
    AMPCVAOI as XGBAMPCVAOI,
    AMPCVAOIConfig as XGBConfig,
    XGBoostLocalSearchPolicy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAILS_CSV = OUTPUT_DIR / "benchmark_results_detailed.csv"
SUMMARY_CSV = OUTPUT_DIR / "benchmark_results_summary.csv"

INSTANCE_SIZES = [30, 40, 50, 60]
INSTANCE_SEEDS = range(1, 11)
SOLVER_SEEDS = range(1, 6)

GENERATIONS = 200
POPULATION_SIZE = 10
MUTATION_RATE = 0.08
LOCAL_SEARCH_MODE = "2opt"


@dataclass(frozen=True)
class PolicyBenchmarkConfig:
    label: str
    family: str
    model_path: Path
    features_path: Path
    metrics_path: Path
    fallback_threshold: float


XGB_IMPROVED = PolicyBenchmarkConfig(
    label="am_pcva_oi_xgboost",
    family="xgboost",
    model_path=PROJECT_ROOT / "artifacts" / "xgboost_improved_model.joblib",
    features_path=PROJECT_ROOT / "artifacts" / "xgboost_feature_columns.json",
    metrics_path=PROJECT_ROOT / "artifacts" / "xgboost_metrics.json",
    fallback_threshold=0.80,
)

LGBM_IMPROVED = PolicyBenchmarkConfig(
    label="am_pcva_oi_lightgbm",
    family="lightgbm",
    model_path=PROJECT_ROOT / "artifacts" / "lightgbm_improved_model.joblib",
    features_path=PROJECT_ROOT / "artifacts" / "lightgbm_feature_columns.json",
    metrics_path=PROJECT_ROOT / "artifacts" / "lightgbm_metrics.json",
    fallback_threshold=0.75,
)

XGB_EFFICIENCY = PolicyBenchmarkConfig(
    label="am_pcva_oi_xgboost_efficiency",
    family="xgboost",
    model_path=PROJECT_ROOT / "artifacts" / "xgboost_efficiency_model.joblib",
    features_path=PROJECT_ROOT / "artifacts" / "xgboost_efficiency_feature_columns.json",
    metrics_path=PROJECT_ROOT / "artifacts" / "xgboost_efficiency_metrics.json",
    fallback_threshold=0.50,
)

LGBM_EFFICIENCY = PolicyBenchmarkConfig(
    label="am_pcva_oi_lightgbm_efficiency",
    family="lightgbm",
    model_path=PROJECT_ROOT / "artifacts" / "lightgbm_efficiency_model.joblib",
    features_path=PROJECT_ROOT / "artifacts" / "lightgbm_efficiency_feature_columns.json",
    metrics_path=PROJECT_ROOT / "artifacts" / "lightgbm_efficiency_metrics.json",
    fallback_threshold=0.50,
)


def load_threshold(metrics_path: Path, fallback_threshold: float) -> float:
    if not metrics_path.exists():
        return fallback_threshold

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    return float(
        metrics.get(
            "recommended_threshold",
            metrics.get("threshold", metrics.get("default_threshold", fallback_threshold)),
        )
    )


def instrument_solver(solver):
    stats = {
        "local_search_opportunities": 0,
        "local_search_skipped": 0,
        "local_search_calls_total": 0,
        "local_search_calls_initialization": 0,
        "local_search_calls_main_loop": 0,
        "local_search_improvements": 0,
        "local_search_total_time_ms": 0.0,
        "local_search_total_delta_cost": 0.0,
    }

    solver._benchmark_stats = stats
    solver._benchmark_ls_stage = "outside"

    original_initialize_population = solver._initialize_population
    original_maybe_apply_local_search = solver._maybe_apply_local_search
    original_apply_local_search = solver._apply_local_search

    def instrumented_initialize_population(self):
        previous_stage = self._benchmark_ls_stage
        self._benchmark_ls_stage = "initialization"
        try:
            return original_initialize_population()
        finally:
            self._benchmark_ls_stage = previous_stage

    def instrumented_maybe_apply_local_search(self, individual, population, generation):
        if generation >= 2:
            self._benchmark_stats["local_search_opportunities"] += 1

        previous_calls = self._benchmark_stats["local_search_calls_total"]
        previous_stage = self._benchmark_ls_stage
        self._benchmark_ls_stage = "main_loop"

        try:
            result = original_maybe_apply_local_search(individual, population, generation)
        finally:
            self._benchmark_ls_stage = previous_stage

        if generation >= 2 and self._benchmark_stats["local_search_calls_total"] == previous_calls:
            self._benchmark_stats["local_search_skipped"] += 1

        return result

    def instrumented_apply_local_search(self, individual):
        before = individual.cost
        t0 = time.perf_counter()
        improved_ind = original_apply_local_search(individual)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        delta = before - improved_ind.cost

        self._benchmark_stats["local_search_calls_total"] += 1
        self._benchmark_stats["local_search_total_time_ms"] += elapsed_ms
        self._benchmark_stats["local_search_total_delta_cost"] += max(delta, 0.0)

        if delta > 1e-12:
            self._benchmark_stats["local_search_improvements"] += 1

        if self._benchmark_ls_stage == "initialization":
            self._benchmark_stats["local_search_calls_initialization"] += 1
        else:
            self._benchmark_stats["local_search_calls_main_loop"] += 1

        return improved_ind

    def get_run_stats(self):
        run_stats = dict(self._benchmark_stats)
        calls = run_stats["local_search_calls_total"]
        opportunities = run_stats["local_search_opportunities"]
        run_stats["local_search_mean_time_ms"] = (
            run_stats["local_search_total_time_ms"] / calls if calls else 0.0
        )
        run_stats["local_search_activation_rate"] = (
            run_stats["local_search_calls_main_loop"] / opportunities if opportunities else 0.0
        )
        return run_stats

    solver._initialize_population = MethodType(instrumented_initialize_population, solver)
    solver._maybe_apply_local_search = MethodType(instrumented_maybe_apply_local_search, solver)
    solver._apply_local_search = MethodType(instrumented_apply_local_search, solver)
    solver.get_run_stats = MethodType(get_run_stats, solver)
    return solver


def build_base_solver(dist, seed):
    config = BaseConfig(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        local_search_mode=LOCAL_SEARCH_MODE,
        seed=seed,
    )
    solver = BaseAMPCVAOI(
        dist=dist,
        config=config,
        policy=None,
        collect_decisions=False,
    )
    return instrument_solver(solver)


def build_xgb_solver(dist, seed, model_path: Path, features_path: Path, threshold: float):
    config = XGBConfig(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        local_search_mode=LOCAL_SEARCH_MODE,
        seed=seed,
    )
    policy = XGBoostLocalSearchPolicy(
        model_path=model_path,
        features_path=features_path,
        threshold=threshold,
    )
    solver = XGBAMPCVAOI(
        dist=dist,
        config=config,
        policy=policy,
        collect_decisions=False,
    )
    return instrument_solver(solver)


def build_lgbm_solver(dist, seed, model_path: Path, features_path: Path, threshold: float):
    config = LGBMConfig(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        local_search_mode=LOCAL_SEARCH_MODE,
        seed=seed,
    )
    policy = LightGBMLocalSearchPolicy(
        model_path=model_path,
        features_path=features_path,
        threshold=threshold,
    )
    solver = LGBMAMPCVAOI(
        dist=dist,
        config=config,
        policy=policy,
        collect_decisions=False,
    )
    return instrument_solver(solver)


def run_solver(label, solver, instance_size, instance_seed, solver_seed):
    t0 = time.perf_counter()
    best = solver.run()
    elapsed = time.perf_counter() - t0
    run_stats = solver.get_run_stats()

    return {
        "approach": label,
        "instance_size": instance_size,
        "instance_seed": instance_seed,
        "solver_seed": solver_seed,
        "best_cost": float(best.cost),
        "runtime_seconds": float(elapsed),
        "local_search_opportunities": int(run_stats["local_search_opportunities"]),
        "local_search_skipped": int(run_stats["local_search_skipped"]),
        "local_search_calls_total": int(run_stats["local_search_calls_total"]),
        "local_search_calls_initialization": int(run_stats["local_search_calls_initialization"]),
        "local_search_calls_main_loop": int(run_stats["local_search_calls_main_loop"]),
        "local_search_improvements": int(run_stats["local_search_improvements"]),
        "local_search_total_time_ms": float(run_stats["local_search_total_time_ms"]),
        "local_search_total_delta_cost": float(run_stats["local_search_total_delta_cost"]),
        "local_search_mean_time_ms": float(run_stats["local_search_mean_time_ms"]),
        "local_search_activation_rate": float(run_stats["local_search_activation_rate"]),
    }


def discover_approaches():
    approaches = [
        {
            "label": "am_pcva_oi_base",
            "builder": build_base_solver,
        }
    ]

    for candidate in [XGB_IMPROVED, LGBM_IMPROVED, XGB_EFFICIENCY, LGBM_EFFICIENCY]:
        if not candidate.model_path.exists() or not candidate.features_path.exists():
            print(f"[SKIP] Artefatos ausentes para {candidate.label}")
            continue

        threshold = load_threshold(candidate.metrics_path, candidate.fallback_threshold)
        print(
            f"[INFO] Policy carregada: {candidate.label} | "
            f"threshold={threshold:.2f} | model={candidate.model_path.name}"
        )

        if candidate.family == "xgboost":
            builder = (
                lambda dist, seed, model_path=candidate.model_path,
                features_path=candidate.features_path, threshold=threshold:
                build_xgb_solver(dist, seed, model_path, features_path, threshold)
            )
        else:
            builder = (
                lambda dist, seed, model_path=candidate.model_path,
                features_path=candidate.features_path, threshold=threshold:
                build_lgbm_solver(dist, seed, model_path, features_path, threshold)
            )

        approaches.append(
            {
                "label": candidate.label,
                "builder": builder,
            }
        )

    return approaches


def main():
    results = []
    approaches = discover_approaches()

    total_runs = len(INSTANCE_SIZES) * len(INSTANCE_SEEDS) * len(SOLVER_SEEDS) * len(approaches)
    current = 0

    print("=" * 80)
    print("[START] Benchmark de policies do AM_PCVA-OI")
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
            median_runtime_seconds=("runtime_seconds", "median"),
            std_runtime_seconds=("runtime_seconds", "std"),
            mean_local_search_calls_total=("local_search_calls_total", "mean"),
            mean_local_search_calls_main_loop=("local_search_calls_main_loop", "mean"),
            mean_local_search_calls_initialization=("local_search_calls_initialization", "mean"),
            mean_local_search_skipped=("local_search_skipped", "mean"),
            mean_local_search_improvements=("local_search_improvements", "mean"),
            mean_local_search_activation_rate=("local_search_activation_rate", "mean"),
            mean_local_search_total_time_ms=("local_search_total_time_ms", "mean"),
        )
        .reset_index()
        .sort_values(["mean_best_cost", "mean_runtime_seconds"], ascending=[True, True])
    )

    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print(summary.to_string(index=False))
    print("=" * 80)

    by_size = (
        df.groupby(["approach", "instance_size"])
        .agg(
            runs=("best_cost", "count"),
            mean_best_cost=("best_cost", "mean"),
            median_best_cost=("best_cost", "median"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_local_search_calls_total=("local_search_calls_total", "mean"),
            mean_local_search_calls_main_loop=("local_search_calls_main_loop", "mean"),
            mean_local_search_improvements=("local_search_improvements", "mean"),
            mean_local_search_activation_rate=("local_search_activation_rate", "mean"),
        )
        .reset_index()
        .sort_values(["instance_size", "mean_best_cost"], ascending=[True, True])
    )

    print("\n" + "=" * 80)
    print("[SUMMARY BY INSTANCE SIZE]")
    print(by_size.to_string(index=False))
    print("=" * 80)

    print("\n[DONE]")
    print(f"Detalhado salvo em: {DETAILS_CSV}")
    print(f"Resumo salvo em:    {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
