import time
from pathlib import Path
import pandas as pd
from am_pcva_oi_base import AMPCVAOI, AMPCVAOIConfig, random_euclidean_instance

# =========================
# CONFIGURAÇÃO
# =========================
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "decision_dataset_full.csv"

sizes = [30, 40, 50, 60]
instance_seeds = range(1, 21)
solver_seeds = range(1, 6)

# =========================
# PREPARAÇÃO
# =========================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_records = []

total_runs = len(sizes) * len(instance_seeds) * len(solver_seeds)
current_run = 0

global_start = time.perf_counter()

print("=" * 70)
print("[START] Geração do dataset iniciada")
print(f"[CONFIG] sizes={sizes}")
print(f"[CONFIG] instance_seeds={instance_seeds.start}..{instance_seeds.stop - 1}")
print(f"[CONFIG] solver_seeds={solver_seeds.start}..{solver_seeds.stop - 1}")
print(f"[CONFIG] total de execuções: {total_runs}")
print("=" * 70)

# =========================
# GERAÇÃO
# =========================
for n in sizes:
    for inst_seed in instance_seeds:
        dist = random_euclidean_instance(n=n, seed=inst_seed)

        for solver_seed in solver_seeds:
            current_run += 1
            run_start = time.perf_counter()

            print(
                f"[RUN {current_run}/{total_runs}] "
                f"instance_size={n} | instance_seed={inst_seed} | solver_seed={solver_seed} ...",
                end=""
            )

            config = AMPCVAOIConfig(
                population_size=10,
                generations=200,
                mutation_rate=0.08,
                local_search_mode="2opt",
                seed=solver_seed,
            )

            solver = AMPCVAOI(dist, config=config, collect_decisions=True)
            solver.run()

            run_records = 0

            for rec in solver.decision_records:
                row = rec.__dict__.copy()
                row["instance_size"] = n
                row["instance_seed"] = inst_seed
                row["solver_seed"] = solver_seed
                all_records.append(row)
                run_records += 1

            run_elapsed = time.perf_counter() - run_start

            print(
                f" OK | records={run_records} | "
                f"total_records={len(all_records)} | "
                f"time={run_elapsed:.2f}s"
            )

# =========================
# CONSOLIDAÇÃO
# =========================
print("\n[INFO] Convertendo registros para DataFrame...")
dataset = pd.DataFrame(all_records)

print(f"[INFO] Linhas antes de remover duplicados: {len(dataset)}")
dataset = dataset.drop_duplicates()
print(f"[INFO] Linhas após remover duplicados: {len(dataset)}")

print(f"[INFO] Salvando dataset em: {OUTPUT_FILE}")
dataset.to_csv(OUTPUT_FILE, index=False)

global_elapsed = time.perf_counter() - global_start

print("=" * 70)
print("[DONE] Dataset gerado com sucesso")
print(f"[DONE] Arquivo: {OUTPUT_FILE}")
print(f"[DONE] Linhas finais: {len(dataset)}")
print(f"[DONE] Tempo total: {global_elapsed:.2f}s")
print("=" * 70)