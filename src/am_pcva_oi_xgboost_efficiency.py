from __future__ import annotations

import json
from pathlib import Path

from am_pcva_oi_xgboost import (
    AMPCVAOI,
    AMPCVAOIConfig,
    XGBoostLocalSearchPolicy,
    random_euclidean_instance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "xgboost_efficiency_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "artifacts" / "xgboost_efficiency_feature_columns.json"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "xgboost_efficiency_metrics.json"
DEFAULT_THRESHOLD = 0.50


def load_threshold(metrics_path: Path, fallback: float) -> float:
    if not metrics_path.exists():
        return fallback

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    return float(
        metrics.get(
            "recommended_threshold",
            metrics.get("threshold", metrics.get("default_threshold", fallback)),
        )
    )


def main() -> None:
    threshold = load_threshold(METRICS_PATH, DEFAULT_THRESHOLD)
    dist = random_euclidean_instance(n=50, seed=7)

    config = AMPCVAOIConfig(
        population_size=10,
        generations=200,
        mutation_rate=0.08,
        local_search_mode="2opt",
        seed=7,
    )

    policy = XGBoostLocalSearchPolicy(
        model_path=MODEL_PATH,
        features_path=FEATURES_PATH,
        threshold=threshold,
    )

    solver = AMPCVAOI(
        dist=dist,
        config=config,
        policy=policy,
        collect_decisions=False,
    )

    best = solver.run()

    print("=" * 70)
    print("[RESULTADO FINAL - AM_PCVA_OI + XGBOOST EFFICIENCY]")
    print(f"Threshold usado: {threshold:.2f}")
    print(f"Melhor custo: {best.cost:.6f}")
    print(f"Melhor tour: {best.tour}")
    print("=" * 70)


if __name__ == "__main__":
    main()
