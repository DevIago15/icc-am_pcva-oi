from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = PROJECT_ROOT / "data" / "decision_dataset_full.csv"
OUTPUT_DATASET = PROJECT_ROOT / "data" / "decision_dataset_efficiency.csv"
SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "efficiency_target_summary.json"

REQUIRED_COLUMNS = {
    "improved",
    "delta_cost",
    "local_search_applied",
    "local_search_time_ms",
}


def main() -> None:
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {SOURCE_DATASET}")

    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("[START] Preparacao do dataset derivado por eficiencia")
    print(f"[INFO] Fonte: {SOURCE_DATASET}")
    print("=" * 70)

    df = pd.read_csv(SOURCE_DATASET)
    df.columns = df.columns.str.strip().str.lower()

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing_columns}")

    before_drop = len(df)
    df = df.drop_duplicates().copy()
    after_drop = len(df)

    if after_drop != before_drop:
        print(f"[INFO] Duplicados removidos: {before_drop - after_drop}")

    df["efficiency"] = 0.0

    valid_runtime_mask = df["local_search_time_ms"] > 0
    applied_mask = df["local_search_applied"].astype(int) == 1
    efficiency_mask = applied_mask & valid_runtime_mask

    df.loc[efficiency_mask, "efficiency"] = (
        df.loc[efficiency_mask, "delta_cost"] / df.loc[efficiency_mask, "local_search_time_ms"]
    )

    improved_mask = df["improved"].astype(int) == 1
    positive_efficiency = df.loc[improved_mask & (df["efficiency"] > 0), "efficiency"]

    if positive_efficiency.empty:
        raise ValueError("Nao ha casos positivos com eficiencia > 0 para definir o limiar.")

    efficiency_threshold = float(positive_efficiency.median())

    df["target_efficiency"] = (
        improved_mask & (df["efficiency"] >= efficiency_threshold)
    ).astype(int)

    df.to_csv(OUTPUT_DATASET, index=False)

    summary = {
        "source_dataset": str(SOURCE_DATASET),
        "output_dataset": str(OUTPUT_DATASET),
        "rows": int(len(df)),
        "improved_positive_count": int(improved_mask.sum()),
        "improved_positive_rate": float(improved_mask.mean()),
        "efficiency_threshold_median_positive": efficiency_threshold,
        "target_efficiency_positive_count": int(df["target_efficiency"].sum()),
        "target_efficiency_positive_rate": float(df["target_efficiency"].mean()),
        "local_search_applied_count": int(applied_mask.sum()),
        "valid_runtime_count": int(valid_runtime_mask.sum()),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[SUMMARY]")
    print(f"Linhas finais: {len(df)}")
    print(f"Mediana da eficiencia entre positivos reais: {efficiency_threshold:.8f}")
    print(f"Positivos em improved: {int(improved_mask.sum())} ({improved_mask.mean():.4%})")
    print(
        "[INFO] Positivos em target_efficiency: "
        f"{int(df['target_efficiency'].sum())} ({df['target_efficiency'].mean():.4%})"
    )
    print(f"[DONE] Dataset salvo em: {OUTPUT_DATASET}")
    print(f"[DONE] Resumo salvo em:  {SUMMARY_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
