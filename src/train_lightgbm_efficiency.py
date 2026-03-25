from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "decision_dataset_efficiency.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = OUTPUT_DIR / "lightgbm_efficiency_model.joblib"
FEATURES_PATH = OUTPUT_DIR / "lightgbm_efficiency_feature_columns.json"
METRICS_PATH = OUTPUT_DIR / "lightgbm_efficiency_metrics.json"
IMPORTANCE_CSV_PATH = OUTPUT_DIR / "lightgbm_efficiency_feature_importance.csv"
PR_CURVE_PATH = OUTPUT_DIR / "lightgbm_efficiency_pr_curve.png"
THRESHOLD_SEARCH_CSV_PATH = OUTPUT_DIR / "lightgbm_efficiency_threshold_search.csv"

TARGET_COL = "target_efficiency"

DROP_COLS = [
    "target_efficiency",
    "efficiency",
    "improved",
    "delta_cost",
    "local_search_time_ms",
    "local_search_applied",
    "instance_seed",
    "solver_seed",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42
BASELINE_THRESHOLD = 0.50


def safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def evaluate_threshold(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_positive_count": int(y_pred.sum()),
        "predicted_positive_rate": float(y_pred.mean()),
    }


def find_recommended_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        rows.append(evaluate_threshold(y_true, y_prob, float(threshold)))

    threshold_df = pd.DataFrame(rows)
    valid_rows = threshold_df[threshold_df["predicted_positive_count"] > 0].copy()

    if valid_rows.empty:
        return BASELINE_THRESHOLD, threshold_df

    best_row = valid_rows.sort_values(
        ["f1_score", "precision", "predicted_positive_rate", "threshold"],
        ascending=[False, False, True, False],
    ).iloc[0]
    return float(best_row["threshold"]), threshold_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {DATASET_PATH}")

    print("=" * 70)
    print("[START] Treinamento LightGBM para target de eficiencia")
    print(f"[INFO] Dataset: {DATASET_PATH}")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip().str.lower()

    if TARGET_COL not in df.columns:
        raise ValueError(f"A coluna alvo '{TARGET_COL}' nao existe no dataset.")

    missing_drop_cols = [col for col in DROP_COLS if col not in df.columns]
    if missing_drop_cols:
        raise ValueError(f"Colunas esperadas nao encontradas no dataset: {missing_drop_cols}")

    before_drop = len(df)
    df = df.drop_duplicates()
    after_drop = len(df)

    if after_drop != before_drop:
        print(f"[INFO] Duplicados removidos: {before_drop - after_drop}")

    X = df.drop(columns=DROP_COLS).copy()
    y = df[TARGET_COL].astype(int).copy()

    constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    if constant_cols:
        print(f"[INFO] Colunas constantes removidas: {constant_cols}")
        X = X.drop(columns=constant_cols)

    feature_columns = X.columns.tolist()

    groups = (
        df["instance_size"].astype(str)
        + "_"
        + df["instance_seed"].astype(str)
        + "_"
        + df["solver_seed"].astype(str)
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    train_pos = int(y_train.sum())
    train_neg = int((y_train == 0).sum())
    scale_pos_weight = train_neg / max(train_pos, 1)

    print(f"[INFO] Features usadas: {len(feature_columns)}")
    print(f"[INFO] Tamanho treino: {len(X_train)}")
    print(f"[INFO] Tamanho teste: {len(X_test)}")
    print(f"[INFO] Positivos treino: {train_pos}")
    print(f"[INFO] Negativos treino: {train_neg}")
    print(f"[INFO] scale_pos_weight: {scale_pos_weight:.4f}")

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    print("[INFO] Treinando modelo...")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate_threshold(y_test, y_prob, BASELINE_THRESHOLD)
    recommended_threshold, threshold_df = find_recommended_threshold(y_test, y_prob)
    recommended_metrics = evaluate_threshold(y_test, y_prob, recommended_threshold)

    roc_auc = safe_roc_auc(y_test, y_prob)
    pr_auc = safe_pr_auc(y_test, y_prob)
    y_pred_recommended = (y_prob >= recommended_threshold).astype(int)

    print("=" * 70)
    print("[RESULTADOS - BASELINE]")
    print(f"Threshold: {BASELINE_THRESHOLD:.2f}")
    print(f"Precision: {baseline_metrics['precision']:.4f}")
    print(f"Recall:    {baseline_metrics['recall']:.4f}")
    print(f"F1-score:  {baseline_metrics['f1_score']:.4f}")
    print("=" * 70)

    print("[RESULTADOS - RECOMENDADO]")
    print(f"Threshold: {recommended_threshold:.2f}")
    print(f"Precision: {recommended_metrics['precision']:.4f}")
    print(f"Recall:    {recommended_metrics['recall']:.4f}")
    print(f"F1-score:  {recommended_metrics['f1_score']:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC:   {roc_auc:.4f}")
    if pr_auc is not None:
        print(f"PR-AUC:    {pr_auc:.4f}")
    print("=" * 70)

    print("\n[CLASSIFICATION REPORT - RECOMENDADO]")
    print(classification_report(y_test, y_pred_recommended, digits=4, zero_division=0))

    print("[CONFUSION MATRIX - RECOMENDADO]")
    print(confusion_matrix(y_test, y_pred_recommended))

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 5))
    plt.plot(recall_curve, precision_curve)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall - LightGBM (Efficiency)")
    plt.tight_layout()
    plt.savefig(PR_CURVE_PATH, dpi=150)
    plt.close()

    threshold_df.to_csv(THRESHOLD_SEARCH_CSV_PATH, index=False)
    importance_df.to_csv(IMPORTANCE_CSV_PATH, index=False)
    joblib.dump(model, MODEL_PATH)

    with FEATURES_PATH.open("w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)

    metrics = {
        "target_column": TARGET_COL,
        "baseline_threshold": float(BASELINE_THRESHOLD),
        "recommended_threshold": float(recommended_threshold),
        "baseline_metrics": baseline_metrics,
        "recommended_metrics": recommended_metrics,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "positives_train": train_pos,
        "negatives_train": train_neg,
        "scale_pos_weight": float(scale_pos_weight),
        "dropped_constant_columns": constant_cols,
        "used_features": feature_columns,
        "groups_train": int(groups.iloc[train_idx].nunique()),
        "groups_test": int(groups.iloc[test_idx].nunique()),
        "dropped_columns": DROP_COLS,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[DONE] Modelo salvo em:", MODEL_PATH)
    print("[DONE] Features salvas em:", FEATURES_PATH)
    print("[DONE] Metricas salvas em:", METRICS_PATH)
    print("[DONE] Importancia salva em:", IMPORTANCE_CSV_PATH)
    print("[DONE] Curva PR salva em:", PR_CURVE_PATH)
    print("[DONE] Busca de threshold salva em:", THRESHOLD_SEARCH_CSV_PATH)


if __name__ == "__main__":
    main()
