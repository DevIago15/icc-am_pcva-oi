from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from am_pcva_oi_base import (
    AMPCVAOI,
    AMPCVAOIConfig,
    LocalSearchPolicy,
    random_euclidean_instance,
)


class LightGBMLocalSearchPolicy(LocalSearchPolicy):
    def __init__(
        self,
        model_path: str | Path,
        features_path: str | Path,
        threshold: float = 0.75,
    ):
        self.model = joblib.load(model_path)

        with Path(features_path).open("r", encoding="utf-8") as f:
            self.feature_names = json.load(f)

        self.threshold = threshold

    def should_apply(self, features: dict) -> bool:
        row = {name: features.get(name, 0.0) for name in self.feature_names}
        X = pd.DataFrame([row], columns=self.feature_names)
        prob = float(self.model.predict_proba(X)[0][1])
        return prob >= self.threshold


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dist = random_euclidean_instance(n=50, seed=7)

    config = AMPCVAOIConfig(
        population_size=10,
        generations=200,
        mutation_rate=0.08,
        local_search_mode="2opt",
        seed=7,
    )

    policy = LightGBMLocalSearchPolicy(
        model_path=project_root / "artifacts" / "lightgbm_improved_model.joblib",
        features_path=project_root / "artifacts" / "lightgbm_feature_columns.json",
        threshold=0.75,
    )

    solver = AMPCVAOI(
        dist=dist,
        config=config,
        policy=policy,
        collect_decisions=False,
    )

    best = solver.run()

    print("=" * 70)
    print("[RESULTADO FINAL - AM_PCVA_OI + LIGHTGBM]")
    print(f"Melhor custo: {best.cost:.6f}")
    print(f"Melhor tour: {best.tour}")
    print("=" * 70)


if __name__ == "__main__":
    main()
