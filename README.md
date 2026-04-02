# AM-PCVA-OI + Machine Learning para PCV

Projeto experimental de Algoritmo Memetico para o Problema do Caixeiro Viajante (PCV / TSP), com uma trilha complementar de Machine Learning para decidir quando aplicar busca local.

O estado atual do projeto e o benchmark mais recente indicam que a base sem ML deve ser mantida como referencia principal. As policies com XGBoost e LightGBM continuam no repositorio como linha experimental, mas nao como nova base do solver. A partir dessa base unica, o projeto agora tambem possui uma trilha hibrida com Grover, preparada para comparacoes futuras entre execucao classica e quantica.

## Visao geral

O projeto implementa:

- representacao por permutacao
- crossover `OX1`
- mutacao `ISM`
- busca local `2-opt`
- busca local hibrida `grover_2opt`
- coleta de decisoes durante a execucao
- treino de policies com `XGBoost` e `LightGBM`
- benchmark comparativo entre solver base e variantes com ML
- benchmark dedicado para comparar `2-opt` classico com a versao hibrida baseada em Grover

A pergunta central do trabalho foi:

> um modelo supervisionado consegue aprender quando vale a pena aplicar busca local em um algoritmo memetico?

## Decisao atual do projeto

A escolha atual do projeto e seguir com o solver base, sem ML, como implementacao principal.

Motivos:

- no benchmark corrigido e justo, o `am_pcva_oi_base` obteve o melhor custo medio final
- as policies com ML reduziram chamadas de busca local, mas nao melhoraram o custo final
- as variantes `improved` ficaram piores em custo e tambem mais lentas que o baseline
- as variantes `efficiency` reduziram tempo, mas perderam qualidade demais
- antes da etapa com modulo quantico, faz mais sentido consolidar uma unica base classica confiavel

Em outras palavras: o ML foi util como investigacao cientifica, mas nao se mostrou vantajoso o suficiente para substituir a base classica. Por isso, a extensao com Grover foi acoplada diretamente ao solver base, e nao a uma variante com ML.

## Resultado do benchmark corrigido

O benchmark mais recente foi executado apos consolidar o solver em um nucleo unico compartilhado. Isso eliminou a assimetria anterior na inicializacao da populacao e tornou a comparacao entre abordagens estruturalmente justa.

Resumo agregado:

| Abordagem | Runs | Mean best cost | Mean runtime (s) | Mean LS calls total | Mean LS calls main loop | Mean LS activation rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `am_pcva_oi_base` | 200 | 5.607753 | 7.179107 | 1040.800 | 1030.800 | 1.000000 |
| `am_pcva_oi_lightgbm` | 200 | 5.627831 | 7.933481 | 100.855 | 90.855 | 0.091065 |
| `am_pcva_oi_xgboost` | 200 | 5.633813 | 8.476708 | 94.530 | 84.530 | 0.082710 |
| `am_pcva_oi_xgboost_efficiency` | 200 | 5.733702 | 5.620585 | 58.955 | 48.955 | 0.051401 |
| `am_pcva_oi_lightgbm_efficiency` | 200 | 5.734233 | 4.933695 | 60.895 | 50.895 | 0.052631 |

Leitura objetiva:

- `base` foi o melhor em custo medio final
- `lightgbm` e `xgboost` cortaram muitas chamadas de busca local, mas ficaram piores em custo e ainda mais lentos que o baseline
- `xgboost_efficiency` e `lightgbm_efficiency` reduziram runtime, mas perderam qualidade de solucao de forma relevante

Conclusao experimental:

- se o foco principal for qualidade da solucao, o baseline e a melhor escolha
- se o foco for reduzir tempo, a trilha `efficiency` ainda nao compensa a perda de qualidade
- portanto, a linha principal do projeto permanece sendo o solver base sem ML

## Estrutura atual

```text
pcv_memetic_ml/
|-- artifacts/
|   |-- benchmark_results_detailed.csv
|   |-- benchmark_results_summary.csv
|   |-- benchmark_grover_detailed.csv
|   |-- benchmark_grover_summary.csv
|   |-- efficiency_target_summary.json
|   |-- lightgbm_feature_columns.json
|   |-- lightgbm_improved_model.joblib
|   |-- lightgbm_metrics.json
|   |-- lightgbm_efficiency_feature_columns.json
|   |-- lightgbm_efficiency_model.joblib
|   |-- lightgbm_efficiency_metrics.json
|   |-- xgboost_feature_columns.json
|   |-- xgboost_improved_model.joblib
|   |-- xgboost_metrics.json
|   |-- xgboost_efficiency_feature_columns.json
|   |-- xgboost_efficiency_model.joblib
|   |-- xgboost_efficiency_metrics.json
|-- data/
|   |-- decision_dataset.csv
|   |-- decision_dataset_full.csv
|   |-- decision_dataset_efficiency.csv
|-- src/
|   |-- am_pcva_oi_base.py
|   |-- am_pcva_oi_grover.py
|   |-- am_pcva_oi_lightgbm.py
|   |-- am_pcva_oi_lightgbm_efficiency.py
|   |-- am_pcva_oi_xgboost.py
|   |-- am_pcva_oi_xgboost_efficiency.py
|   |-- benchmark_grover_backends.py
|   |-- benchmark_policies.py
|   |-- generate_dataset.py
|   |-- prepare_efficiency_dataset.py
|   |-- train_lightgbm.py
|   |-- train_lightgbm_efficiency.py
|   |-- train_xgboost.py
|   |-- train_xgboost_efficiency.py
|   |-- valid_decision_dataset.py
|-- requirements.txt
`-- README.md
```

## Arquitetura atual

Hoje existe uma base unica para o solver:

- [src/am_pcva_oi_base.py](/c:/Users/Lagoa/OneDrive/Área%20de%20Trabalho/pes/facul/icc/pcv_memetic_ml/src/am_pcva_oi_base.py)

Esse arquivo concentra:

- o nucleo do algoritmo memetico
- a inicializacao da populacao
- os operadores geneticos
- a busca local
- o modo hibrido `grover_2opt`
- a coleta de decisoes
- as policies genericas
- os backends de busca Grover

Os arquivos abaixo sao apenas wrappers de policy sobre o mesmo nucleo:

- [src/am_pcva_oi_xgboost.py](/c:/Users/Lagoa/OneDrive/Área%20de%20Trabalho/pes/facul/icc/pcv_memetic_ml/src/am_pcva_oi_xgboost.py)
- [src/am_pcva_oi_lightgbm.py](/c:/Users/Lagoa/OneDrive/Área%20de%20Trabalho/pes/facul/icc/pcv_memetic_ml/src/am_pcva_oi_lightgbm.py)
- [src/am_pcva_oi_xgboost_efficiency.py](/c:/Users/Lagoa/OneDrive/Área%20de%20Trabalho/pes/facul/icc/pcv_memetic_ml/src/am_pcva_oi_xgboost_efficiency.py)
- [src/am_pcva_oi_lightgbm_efficiency.py](/c:/Users/Lagoa/OneDrive/Área%20de%20Trabalho/pes/facul/icc/pcv_memetic_ml/src/am_pcva_oi_lightgbm_efficiency.py)

Isso foi feito para evitar drift entre implementacoes e preparar o projeto para a futura extensao com modulo quantico.

## Trilha hibrida com Grover

O projeto agora inclui uma extensao hibrida do solver base usando Grover como subrotina de busca sobre um conjunto de movimentos candidatos de `2-opt`.

Ideia geral:

- o solver continua sendo o `AMPCVAOI` compartilhado
- a busca local pode operar no modo tradicional `2opt`
- ou no modo `grover_2opt`
- no modo hibrido, o solver monta um pool de movimentos `2-opt` candidatos
- um backend Grover escolhe um movimento melhorante dentro desse pool

Backends atualmente suportados:

- `ClassicalGroverSearchBackend`
- `QiskitGroverSearchBackend`

Objetivo dessa trilha:

- manter uma unica base classica confiavel
- desenvolver a versao hibrida sobre o mesmo nucleo do solver
- permitir comparacao futura entre execucao em ambiente classico e ambiente quantico

No estado atual:

- o backend classico ja esta funcional
- o backend Qiskit e opcional e depende da instalacao das bibliotecas quanticas
- a comparacao com hardware quantico real fica preparada como proximo passo, nao como etapa concluida

## Pipeline experimental

Fluxo atual:

```text
1. Executar o solver base com coleta de decisoes
   ->
2. Gerar dataset completo
   ->
3. Validar dataset
   ->
4. Treinar models para target improved
   ->
5. Gerar dataset derivado por efficiency
   ->
6. Treinar models para target efficiency
   ->
7. Rodar benchmark comparativo
   ->
8. Consolidar o solver base como linha principal
   ->
9. Estender essa base para a versao hibrida com Grover
   ->
10. Comparar execucao classica e quantica no futuro
```

## Datasets e targets

### Dataset principal

Gerado por:

- `python src/am_pcva_oi_base.py`
- `python src/generate_dataset.py`

Arquivos:

- `data/decision_dataset.csv`
- `data/decision_dataset_full.csv`

### Target original

`improved`

Interpretacao:

- `1`: a busca local melhorou a solucao
- `0`: a busca local nao melhorou a solucao

### Dataset derivado por eficiencia

Gerado por:

- `python src/prepare_efficiency_dataset.py`

Arquivo:

- `data/decision_dataset_efficiency.csv`

Esse dataset adiciona:

- `efficiency = delta_cost / local_search_time_ms`
- `target_efficiency = 1` apenas quando houve melhora real e a eficiencia ficou acima do limiar definido pela mediana da eficiencia entre os casos positivos

## Features usadas pelos modelos

As policies aprendem a partir destas features:

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

Colunas removidas do treino por vazamento ou identidade experimental:

- `delta_cost`
- `local_search_time_ms`
- `local_search_applied`
- `instance_seed`
- `solver_seed`

Na trilha de eficiencia, tambem sao removidas:

- `efficiency`
- `improved`

## Como executar

### 1. Criar e ativar ambiente virtual

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Observacao:

- `qiskit` e `qiskit-algorithms` sao dependencias da trilha Grover
- se voce quiser trabalhar apenas com a parte classica, a base do solver continua utilizavel sem acionar o backend quantico

### 3. Gerar dataset base

```bash
python src/am_pcva_oi_base.py
python src/generate_dataset.py
```

### 4. Validar dataset

```bash
python src/valid_decision_dataset.py
```

### 5. Treinar modelos da trilha `improved`

```bash
python src/train_xgboost.py
python src/train_lightgbm.py
```

### 6. Gerar e treinar trilha `efficiency`

```bash
python src/prepare_efficiency_dataset.py
python src/train_xgboost_efficiency.py
python src/train_lightgbm_efficiency.py
```

### 7. Rodar benchmark

```bash
python src/benchmark_policies.py
```

### 8. Rodar a versao hibrida com Grover

```bash
python src/am_pcva_oi_grover.py
```

### 9. Rodar benchmark da trilha Grover

```bash
python src/benchmark_grover_backends.py
```

## O que cada script faz

### `src/am_pcva_oi_base.py`

Nucleo compartilhado do solver. Quando executado diretamente, gera um dataset inicial de decisoes usando policy exploratoria. Tambem concentra a extensao hibrida `grover_2opt` e os backends de busca Grover.

### `src/am_pcva_oi_grover.py`

Executa a versao hibrida do solver base com `local_search_mode="grover_2opt"`.

### `src/generate_dataset.py`

Expande a coleta para multiplas instancias e multiplas seeds, produzindo a base consolidada para treino.

### `src/prepare_efficiency_dataset.py`

Deriva o dataset da trilha de eficiencia a partir do dataset consolidado.

### `src/train_xgboost.py`

Treina XGBoost para o target `improved`.

### `src/train_lightgbm.py`

Treina LightGBM para o target `improved`.

### `src/train_xgboost_efficiency.py`

Treina XGBoost para o target `target_efficiency`.

### `src/train_lightgbm_efficiency.py`

Treina LightGBM para o target `target_efficiency`.

### `src/am_pcva_oi_xgboost.py`

Executa o solver base com policy aprendida por XGBoost.

### `src/am_pcva_oi_lightgbm.py`

Executa o solver base com policy aprendida por LightGBM.

### `src/am_pcva_oi_xgboost_efficiency.py`

Executa o solver base com policy de eficiencia aprendida por XGBoost.

### `src/am_pcva_oi_lightgbm_efficiency.py`

Executa o solver base com policy de eficiencia aprendida por LightGBM.

### `src/benchmark_policies.py`

Executa benchmark comparativo entre:

- `am_pcva_oi_base`
- `am_pcva_oi_xgboost`
- `am_pcva_oi_lightgbm`
- `am_pcva_oi_xgboost_efficiency`
- `am_pcva_oi_lightgbm_efficiency`

E registra, alem de custo e tempo:

- numero de oportunidades de busca local
- numero de chamadas reais de busca local
- chamadas na inicializacao
- chamadas no loop principal
- taxa de ativacao de busca local
- total de melhorias obtidas por busca local
- tempo total gasto em busca local

### `src/benchmark_grover_backends.py`

Executa benchmark especifico entre:

- `am_pcva_oi_base_2opt`
- `am_pcva_oi_grover_classical`
- `am_pcva_oi_grover_qiskit_statevector` quando `qiskit` estiver disponivel

E registra:

- custo final
- tempo total
- numero de chamadas ao backend Grover
- numero de sucessos do backend Grover
- tamanho medio do pool de candidatos
- tempo total e medio do backend Grover

## Escolha da base sem ML

A escolha atual do projeto e seguir com o solver base sem ML por tres razoes centrais:

1. Ele foi a melhor abordagem em qualidade de solucao no benchmark corrigido.
2. Ele evita o overhead de inferencia dos modelos.
3. Ele fornece uma base classica unica, limpa e estavel para a futura integracao com o modulo quantico.

Isso significa que:

- o solver base e a referencia principal do projeto
- as policies de ML permanecem como trilha secundaria de pesquisa
- qualquer futura extensao quantica deve partir do nucleo compartilhado em `src/am_pcva_oi_base.py`
- a versao com Grover e uma extensao direta da base oficial, e nao uma nova base paralela

## Estado atual do projeto

Neste momento, o projeto esta concentrado em:

- consolidar a base classica do algoritmo memetico
- manter a trilha de ML como linha experimental
- preparar o codigo para uma proxima etapa com modulo quantico
- estruturar uma versao hibrida com Grover sobre a base classica

Ainda nao fazem parte da implementacao atual:

- execucao em hardware quantico real
- avaliacao experimental completa entre backend classico e backend quantico real
- pipeline classico-quantico final consolidado

## Proximos passos

- manter o solver base como baseline oficial
- usar a base compartilhada como ponto de extensao para o modulo quantico
- amadurecer a trilha `grover_2opt`
- instalar e validar o backend Qiskit no ambiente
- comparar `grover_2opt` com backend classico versus backend quantico
- testar instancias reais da TSPLIB
- revisar se alguma policy de ML ainda pode ser aproveitada como heuristica auxiliar, sem substituir a base

## Contexto

Este projeto foi desenvolvido no contexto de Iniciacao Cientifica, conectando:

- metaheuristicas evolutivas
- otimizacao combinatoria
- aprendizado supervisionado
- policies adaptativas de busca local
- futura extensao para computacao quantica
