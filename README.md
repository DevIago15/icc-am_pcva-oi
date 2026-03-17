# 🧠 AM-PCVA-OI + Machine Learning para PCV

Projeto experimental de **Algoritmo Memético** para o **Problema do Caixeiro Viajante (PCV / TSP)** com integração de **Machine Learning** para aprender **quando aplicar busca local** de forma mais inteligente.

Hoje o foco do repositório está em duas frentes:

- ⚙️ construção e refinamento do algoritmo memético
- 🤖 treinamento de modelos de ML para decisão de busca local

Essa é apenas a ponta do iceberg. A ideia futura é conectar **módulos quânticos** para melhorar a execução do algoritmo, mas essa etapa **ainda não está pronta**. O projeto está atualmente na fase de **criação do algoritmo e da camada de ML**.

---

## ✨ Visão geral

O projeto implementa uma base inspirada no **AM-PCVA-OI** com:

- representação por permutação
- cruzamento `OX1`
- mutação `ISM`
- busca local `2-opt`
- coleta de decisões durante a execução
- políticas de decisão baseadas em `XGBoost` e `LightGBM`

A pergunta central é:

> **um modelo supervisionado consegue aprender quando vale a pena aplicar busca local em um algoritmo memético?**

---

## 🧪 Pipeline real do projeto

O fluxo atual acontece em duas etapas de dataset:

```text
1. Executar o AM-PCVA-OI base
   ↓
2. Gerar um dataset simples inicial
   ↓
3. Expandir/melhorar esse dataset com múltiplas execuções
   ↓
4. Validar os dados
   ↓
5. Treinar XGBoost e LightGBM
   ↓
6. Usar os modelos como políticas de busca local
   ↓
7. Comparar base vs XGBoost vs LightGBM
```

Em termos práticos:

- `src/am_pcva_oi_base.py` gera um **dataset simples inicial**
- `src/generate_dataset.py` gera a versão **mais robusta/completa** do dataset para treino

---

## 📁 Estrutura do projeto

```text
pcv_memetic_ml/
├── artifacts/
│   ├── lightgbm_feature_columns.json
│   ├── lightgbm_feature_importance.csv
│   ├── lightgbm_improved_model.joblib
│   ├── lightgbm_metrics.json
│   ├── lightgbm_pr_curve.png
│   ├── xgboost_feature_columns.json
│   ├── xgboost_feature_importance.csv
│   ├── xgboost_improved_model.joblib
│   ├── xgboost_metrics.json
│   └── xgboost_pr_curve.png
├── data/
│   ├── decision_dataset.csv
│   └── decision_dataset_full.csv
├── notebooks/
├── src/
│   ├── am_pcva_oi_base.py
│   ├── am_pcva_oi_lightgbm.py
│   ├── am_pcva_oi_xgboost.py
│   ├── benchmark_policies.py
│   ├── generate_dataset.py
│   ├── train_lightgbm.py
│   ├── train_xgboost.py
│   └── valid_decision_dataset.py
├── requirements.txt
└── README.md
```

---

## 🚀 Como executar

### 1. Criar e ativar ambiente virtual

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Gerar o dataset simples inicial

Esse passo usa o solver base e salva um dataset menor, útil como ponto de partida.

```bash
python src/am_pcva_oi_base.py
```

Saída esperada:

```text
data/decision_dataset.csv
```

### 4. Gerar o dataset completo

Depois do dataset inicial, você expande a base com múltiplas instâncias e múltiplas seeds:

```bash
python src/generate_dataset.py
```

Saída esperada:

```text
data/decision_dataset_full.csv
```

### 5. Validar o dataset

```bash
python src/valid_decision_dataset.py
```

### 6. Treinar os modelos

```bash
python src/train_xgboost.py
python src/train_lightgbm.py
```

### 7. Rodar o benchmark

```bash
python src/benchmark_policies.py
```

---

## 🧭 O que cada script faz

### `src/am_pcva_oi_base.py`

É a base do algoritmo memético e também o primeiro passo de geração de dados.

Quando executado diretamente, ele:

- cria uma instância euclidiana do PCV
- roda o algoritmo base
- coleta decisões de busca local
- exporta um dataset simples inicial

Saída:

```text
data/decision_dataset.csv
```

### `src/generate_dataset.py`

É a etapa de **expansão/melhoria do dataset**.

Ele executa múltiplas combinações de:

- tamanhos de instância
- seeds da instância
- seeds do solver

Objetivo:

- sair de um dataset simples inicial
- gerar uma base mais robusta para treino supervisionado

Saída:

```text
data/decision_dataset_full.csv
```

Configuração atual:

- tamanhos de instância: `30, 40, 50, 60`
- seeds de instância: `1..20`
- seeds do solver: `1..5`

### `src/valid_decision_dataset.py`

Faz uma checagem exploratória da base:

- tipos e estatísticas
- valores nulos
- infinitos
- duplicados
- distribuição de `improved`
- correlação entre colunas numéricas

### `src/train_xgboost.py`

Treina um classificador `XGBoost` para prever `improved`.

Artefatos gerados:

- `artifacts/xgboost_improved_model.joblib`
- `artifacts/xgboost_feature_columns.json`
- `artifacts/xgboost_metrics.json`
- `artifacts/xgboost_feature_importance.csv`
- `artifacts/xgboost_pr_curve.png`

### `src/train_lightgbm.py`

Treina um classificador `LightGBM` com a mesma lógica experimental.

Artefatos gerados:

- `artifacts/lightgbm_improved_model.joblib`
- `artifacts/lightgbm_feature_columns.json`
- `artifacts/lightgbm_metrics.json`
- `artifacts/lightgbm_feature_importance.csv`
- `artifacts/lightgbm_pr_curve.png`

### `src/am_pcva_oi_xgboost.py`

Executa o algoritmo usando uma policy baseada no modelo `XGBoost`.

### `src/am_pcva_oi_lightgbm.py`

Executa o algoritmo usando uma policy baseada no modelo `LightGBM`.

### `src/benchmark_policies.py`

Compara três abordagens:

- `am_pcva_oi_base`
- `am_pcva_oi_xgboost`
- `am_pcva_oi_lightgbm`

Saídas:

- `artifacts/benchmark_results_detailed.csv`
- `artifacts/benchmark_results_summary.csv`

---

## 🎯 Variável alvo

A variável alvo do modelo é:

```text
improved
```

Interpretação:

- `improved = 1` → a busca local melhorou a solução
- `improved = 0` → a busca local não trouxe ganho

Isso transforma a decisão de aplicar busca local em um problema de **classificação binária**.

---

## 🧩 Features usadas pelos modelos

As policies aprendem a partir de sinais extraídos do estado do algoritmo:

```text
generation
individual_rank
individual_cost
best_cost
worst_cost
mean_cost
std_cost
normalized_cost
relative_gap_to_best
age
stagnation
unique_edges_ratio
mean_edge_cost
max_edge_cost
instance_size
```

Colunas removidas no treino por vazamento ou identidade experimental:

- `delta_cost`
- `local_search_time_ms`
- `local_search_applied`
- `instance_seed`
- `solver_seed`

---

## 🧠 Exemplo com o solver base

```python
from src.am_pcva_oi_base import AMPCVAOI, AMPCVAOIConfig, random_euclidean_instance

dist = random_euclidean_instance(n=50, seed=7)

config = AMPCVAOIConfig(
    population_size=10,
    generations=200,
    mutation_rate=0.08,
    local_search_mode="2opt",
    seed=7,
)

solver = AMPCVAOI(
    dist=dist,
    config=config,
    collect_decisions=True,
)

best = solver.run()

print(best.cost)
print(best.tour)
solver.export_decision_dataset("data/decision_dataset.csv")
```

---

## 🌲 Exemplo com policy XGBoost

```python
from src.am_pcva_oi_xgboost import (
    AMPCVAOI,
    AMPCVAOIConfig,
    XGBoostLocalSearchPolicy,
    random_euclidean_instance,
)

dist = random_euclidean_instance(n=50, seed=7)

policy = XGBoostLocalSearchPolicy(
    model_path="artifacts/xgboost_improved_model.joblib",
    features_path="artifacts/xgboost_feature_columns.json",
    threshold=0.80,
)

solver = AMPCVAOI(
    dist=dist,
    config=AMPCVAOIConfig(seed=7, generations=200),
    policy=policy,
    collect_decisions=False,
)

best = solver.run()
print(best.cost)
```

---

## 💡 Exemplo com policy LightGBM

```python
from src.am_pcva_oi_lightgbm import (
    AMPCVAOI,
    AMPCVAOIConfig,
    LightGBMLocalSearchPolicy,
    random_euclidean_instance,
)

dist = random_euclidean_instance(n=50, seed=7)

policy = LightGBMLocalSearchPolicy(
    model_path="artifacts/lightgbm_improved_model.joblib",
    features_path="artifacts/lightgbm_feature_columns.json",
    threshold=0.75,
)

solver = AMPCVAOI(
    dist=dist,
    config=AMPCVAOIConfig(seed=7, generations=200),
    policy=policy,
)

best = solver.run()
print(best.cost)
```

---

## 📊 O que o benchmark mede

Ao final do pipeline, você consegue comparar:

- qualidade da solução final (`best_cost`)
- tempo de execução (`runtime_seconds`)
- impacto das políticas aprendidas
- relação entre custo computacional e ganho de qualidade

Na prática, o benchmark testa se o ML consegue:

- reduzir aplicações desnecessárias de busca local
- manter ou melhorar a qualidade das soluções
- tornar a execução mais eficiente

---

## 🛠️ Dependências principais

O projeto usa principalmente:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `tsplib95`
- `tqdm`

Instalação:

```bash
pip install -r requirements.txt
```

---

## 📌 Estado atual do projeto

Neste momento, o projeto está concentrado em:

- projetar e estabilizar o algoritmo memético
- criar um dataset de decisão confiável
- treinar e validar modelos de ML
- comparar políticas aprendidas contra a abordagem base

Ainda **não** faz parte da implementação atual:

- integração com módulos quânticos
- aceleração quântica do solver
- pipeline híbrido clássico-quântico final

Esses pontos fazem parte da visão futura do projeto.

---

## 🔭 Próximos passos

- refinar ainda mais a geração do dataset
- testar novas features e políticas de decisão
- adicionar novas heurísticas de comparação
- testar instâncias reais da TSPLIB
- incluir módulos quânticos quando a base clássica estiver madura

---

## 👨‍🔬 Contexto

Este projeto foi desenvolvido no contexto de **Iniciação Científica**, conectando:

- metaheurísticas evolutivas
- otimização combinatória
- aprendizado supervisionado
- políticas adaptativas de busca local
- futura extensão para computação quântica
