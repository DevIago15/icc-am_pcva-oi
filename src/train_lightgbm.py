import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# =========================
# CONFIGURAÇÃO
# =========================
DATASET_PATH = Path("data/decision_dataset_full.csv")
OUTPUT_DIR = Path("artifacts")
MODEL_PATH = OUTPUT_DIR / "lightgbm_improved_model.joblib"
FEATURES_PATH = OUTPUT_DIR / "lightgbm_feature_columns.json"
METRICS_PATH = OUTPUT_DIR / "lightgbm_metrics.json"
IMPORTANCE_CSV_PATH = OUTPUT_DIR / "lightgbm_feature_importance.csv"
PR_CURVE_PATH = OUTPUT_DIR / "lightgbm_pr_curve.png"

TARGET_COL = "improved"

# Remover colunas com vazamento ou identidade do experimento
DROP_COLS = [
    "improved",
    "delta_cost",             # vazamento direto
    "local_search_time_ms",   # pós-decisão
    "local_search_applied",   # praticamente entrega parte do alvo
    "instance_seed",          # identidade do experimento
    "solver_seed",            # identidade do experimento
]

TEST_SIZE = 0.20
RANDOM_STATE = 42
THRESHOLD = 0.75

# =========================
# PREPARAÇÃO
# =========================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset não encontrado: {DATASET_PATH}")

print("=" * 70)
print("[START] Treinamento LightGBM com Group Split")
print(f"[INFO] Dataset: {DATASET_PATH}")
print("=" * 70)

# =========================
# LEITURA
# =========================
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip().str.lower()

if TARGET_COL not in df.columns:
    raise ValueError(f"A coluna alvo '{TARGET_COL}' não existe no dataset.")

print(f"[INFO] Linhas originais: {len(df)}")

before_drop = len(df)
df = df.drop_duplicates()
after_drop = len(df)

if after_drop != before_drop:
    print(f"[INFO] Duplicados removidos: {before_drop - after_drop}")

# =========================
# FEATURES E TARGET
# =========================
missing_drop_cols = [c for c in DROP_COLS if c not in df.columns]
if missing_drop_cols:
    raise ValueError(f"Colunas esperadas não encontradas no dataset: {missing_drop_cols}")

X = df.drop(columns=DROP_COLS).copy()
y = df[TARGET_COL].astype(int).copy()

# Remover colunas constantes
constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
if constant_cols:
    print(f"[INFO] Colunas constantes removidas: {constant_cols}")
    X = X.drop(columns=constant_cols)

feature_columns = X.columns.tolist()

print(f"[INFO] Total de features usadas: {len(feature_columns)}")
print(f"[INFO] Features: {feature_columns}")

# =========================
# GROUP SPLIT
# =========================
# Cada execução fica inteira em treino ou teste
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

print(f"[INFO] Tamanho treino: {len(X_train)}")
print(f"[INFO] Tamanho teste: {len(X_test)}")
print(f"[INFO] Grupos treino: {groups.iloc[train_idx].nunique()}")
print(f"[INFO] Grupos teste: {groups.iloc[test_idx].nunique()}")

train_pos = int(y_train.sum())
train_neg = int((y_train == 0).sum())
scale_pos_weight = train_neg / max(train_pos, 1)

print(f"[INFO] Positivos no treino: {train_pos}")
print(f"[INFO] Negativos no treino: {train_neg}")
print(f"[INFO] scale_pos_weight: {scale_pos_weight:.4f}")

print("\n[TRAIN TARGET DISTRIBUTION]")
print(y_train.value_counts(normalize=True))

print("\n[TEST TARGET DISTRIBUTION]")
print(y_test.value_counts(normalize=True))

# =========================
# MODELO
# =========================
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

# =========================
# AVALIAÇÃO
# =========================
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("=" * 70)
print("[RESULTADOS]")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print("=" * 70)

print("\n[CLASSIFICATION REPORT]")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

print("[CONFUSION MATRIX]")
print(confusion_matrix(y_test, y_pred))

print("\n[PRED DISTRIBUTION]")
print(pd.Series(y_pred).value_counts(normalize=True))

print("\n[PROBABILITY STATS]")
print(pd.Series(y_prob).describe())

# =========================
# IMPORTÂNCIA DAS FEATURES
# =========================
importance_df = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }
).sort_values("importance", ascending=False)

print("\n[TOP 20 FEATURES]")
print(importance_df.head(20).to_string(index=False))

# =========================
# CURVA PR
# =========================
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(8, 5))
plt.plot(recall_curve, precision_curve)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall - LightGBM")
plt.tight_layout()
plt.savefig(PR_CURVE_PATH, dpi=150)
plt.close()

# =========================
# SALVAMENTO
# =========================
joblib.dump(model, MODEL_PATH)

with FEATURES_PATH.open("w", encoding="utf-8") as f:
    json.dump(feature_columns, f, ensure_ascii=False, indent=2)

metrics = {
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "threshold": float(THRESHOLD),
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

importance_df.to_csv(IMPORTANCE_CSV_PATH, index=False)

print("=" * 70)
print("[DONE] Artefatos salvos")
print(f"[DONE] Modelo: {MODEL_PATH}")
print(f"[DONE] Features: {FEATURES_PATH}")
print(f"[DONE] Métricas: {METRICS_PATH}")
print(f"[DONE] Importância: {IMPORTANCE_CSV_PATH}")
print(f"[DONE] Curva PR: {PR_CURVE_PATH}")
print("=" * 70)
