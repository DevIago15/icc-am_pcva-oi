import time
from pathlib import Path
import pandas as pd

# ===== IMPORTS DOS SEUS MÓDULOS =====
# Ajuste estes nomes se seus arquivos estiverem diferentes
from am_pcva_oi_base import (
    AMPCVAOI as BaseAMPCVAOI,
    AMPCVAOIConfig as BaseConfig,
    random_euclidean_instance,
)

from am_pcva_oi_xgboost import (
    AMPCVAOI as XGBAMPCVAOI,
    AMPCVAOIConfig as XGBConfig,
    XGBoostLocalSearchPolicy,
)

from am_pcva_oi_lightgbm import (
    AMPCVAOI as LGBMAMPCVAOI,
    AMPCVAOIConfig as LGBMConfig,
    LightGBMLocalSearchPolicy,
)

# =========================
# CONFIGURAÇÃO
# =========================
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAILS_CSV = OUTPUT_DIR / "benchmark_results_detailed.csv"
SUMMARY_CSV = OUTPUT_DIR / "benchmark_results_summary.csv"

INSTANCE_SIZES = [30, 40, 50, 60]
INSTANCE_SEEDS = range(1, 11)   # ajuste se quiser mais
SOLVER_SEEDS = range(1, 6)      # ajuste se quiser mais

GENERATIONS = 200
POPULATION_SIZE = 10
MUTATION_RATE = 0.08
LOCAL_SEARCH_MODE = "2opt"

XGB_MODEL_PATH = "artifacts/xgboost_improved_model.joblib"
XGB_FEATURES_PATH = "artifacts/xgboost_feature_columns.json"
XGB_THRESHOLD = 0.80

LGBM_MODEL_PATH = "artifacts/lightgbm_improved_model.joblib"
LGBM_FEATURES_PATH = "artifacts/lightgbm_feature_columns.json"
LGBM_THRESHOLD = 0.75

# =========================
# FUNÇÕES AUXILIARES
# =========================
def build_base_solver(dist, seed):
    config = BaseConfig(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        local_search_mode=LOCAL_SEARCH_MODE,
        seed=seed,
    )
    return BaseAMPCVAOI(
        dist=dist,
        config=config,
        policy=None,
        collect_decisions=False,
    )

def build_xgb_solver(dist, seed):
    config = XGBConfig(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        local_search_mode=LOCAL_SEARCH_MODE,
        seed=seed,
    )
    policy = XGBoostLocalSearchPolicy(
        model_path=XGB_MODEL_PATH,
        features_path=XGB_FEATURES_PATH,
        threshold=XGB_THRESHOLD,
    )
    return XGBAMPCVAOI(
        dist=dist,
        config=config,
        policy=policy,
        collect_decisions=False,
    )

def build_lgbm_solver(dist, seed):
    config = LGBMConfig(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        local_search_mode=LOCAL_SEARCH_MODE,
        seed=seed,
    )
    policy = LightGBMLocalSearchPolicy(
        model_path=LGBM_MODEL_PATH,
        features_path=LGBM_FEATURES_PATH,
        threshold=LGBM_THRESHOLD,
    )
    return LGBMAMPCVAOI(
        dist=dist,
        config=config,
        policy=policy,
        collect_decisions=False,
    )

def run_solver(label, solver, instance_size, instance_seed, solver_seed):
    t0 = time.perf_counter()
    best = solver.run()
    elapsed = time.perf_counter() - t0

    return {
        "approach": label,
        "instance_size": instance_size,
        "instance_seed": instance_seed,
        "solver_seed": solver_seed,
        "best_cost": float(best.cost),
        "runtime_seconds": float(elapsed),
    }

# =========================
# BENCHMARK
# =========================
def main():
    results = []

    total_runs = len(INSTANCE_SIZES) * len(INSTANCE_SEEDS) * len(SOLVER_SEEDS) * 3
    current = 0

    print("=" * 80)
    print("[START] Benchmark AM_PCVA-OI vs XGBoost vs LightGBM")
    print(f"[INFO] Total de execuções: {total_runs}")
    print("=" * 80)

    for n in INSTANCE_SIZES:
        for inst_seed in INSTANCE_SEEDS:
            dist = random_euclidean_instance(n=n, seed=inst_seed)

            for solver_seed in SOLVER_SEEDS:
                # BASE
                current += 1
                print(f"[{current}/{total_runs}] BASE | n={n} | inst_seed={inst_seed} | solver_seed={solver_seed}")
                solver = build_base_solver(dist, solver_seed)
                results.append(run_solver("am_pcva_oi_base", solver, n, inst_seed, solver_seed))

                # XGBOOST
                current += 1
                print(f"[{current}/{total_runs}] XGBOOST | n={n} | inst_seed={inst_seed} | solver_seed={solver_seed}")
                solver = build_xgb_solver(dist, solver_seed)
                results.append(run_solver("am_pcva_oi_xgboost", solver, n, inst_seed, solver_seed))

                # LIGHTGBM
                current += 1
                print(f"[{current}/{total_runs}] LIGHTGBM | n={n} | inst_seed={inst_seed} | solver_seed={solver_seed}")
                solver = build_lgbm_solver(dist, solver_seed)
                results.append(run_solver("am_pcva_oi_lightgbm", solver, n, inst_seed, solver_seed))

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
        )
        .reset_index()
        .sort_values(["mean_best_cost", "mean_runtime_seconds"], ascending=[True, True])
    )

    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print(summary.to_string(index=False))
    print("=" * 80)

    # Comparação por tamanho de instância
    by_size = (
        df.groupby(["approach", "instance_size"])
        .agg(
            runs=("best_cost", "count"),
            mean_best_cost=("best_cost", "mean"),
            median_best_cost=("best_cost", "median"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
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