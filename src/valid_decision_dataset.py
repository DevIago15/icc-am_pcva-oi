import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# carregar dataset
df = pd.read_csv("data/decision_dataset_full.csv")

print("\n[INFO]")
print(df.info())

print("\n[DESCRIBE]")
print(df.describe())

print("\n[ISNULL]")
print(df.isnull().sum())

print("\n[INFINITOS]")
print(np.isinf(df.select_dtypes(include=[np.number])).sum())

print("\n[DUPLICADOS]")
print(df.duplicated().sum())

print("\n[VALUE_COUNTS - improved]")
print(df['improved'].value_counts())

print("\n[VALUE_COUNTS NORMALIZED - improved]")
print(df['improved'].value_counts(normalize=True))

# validações lógicas
if 'tour_cost' in df.columns:
    print("\n[TOUR_COST NEGATIVO]")
    print((df['tour_cost'] < 0).sum())

if 'gap' in df.columns:
    print("\n[GAP NEGATIVO]")
    print((df['gap'] < 0).sum())

if 'generation' in df.columns:
    print("\n[GENERATION NEGATIVA]")
    print((df['generation'] < 0).sum())

if 'population_diversity' in df.columns:
    print("\n[POPULATION_DIVERSITY FORA DO ESPERADO]")
    print(((df['population_diversity'] < 0) | (df['population_diversity'] > 1)).sum())

# colunas constantes
print("\n[COLUNAS CONSTANTES]")
for col in df.columns:
    if df[col].nunique() <= 1:
        print(f"{col} é constante")

# correlação apenas com colunas numéricas
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.show()

## ideal
# Registros: > 10000
# improved = 1 → 15% – 35%
# sem valores nulos
# sem colunas constantes

## ruim
# Registros < 3000
# improved < 10%
# muitas colunas constantes