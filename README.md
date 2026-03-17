# AM-PCVA-OI + Machine Learning

Implementação experimental de um **Algoritmo Memético inspirado no
AM_PCVA-OI** aplicado ao **Problema do Caixeiro Viajante (PCV)**, com
integração de **Machine Learning (XGBoost e LightGBM)** para aprendizado
de políticas de decisão sobre aplicação de busca local.

O objetivo do projeto é investigar se modelos de aprendizado
supervisionado podem atuar como **módulo de decisão**, aprendendo
**quando e onde aplicar otimizações locais**, reduzindo custo
computacional e melhorando a eficiência do algoritmo memético.

------------------------------------------------------------------------

## Estrutura do Projeto

    pcv_memetic_ml/
    │
    ├── src/
    │   ├── am_pcva_oi_base.py
    │   ├── generate_dataset.py
    │   └── train_model.py
    │
    ├── data/
    │   └── decision_dataset.csv
    │
    ├── notebooks/
    │
    ├── requirements.txt
    │
    └── README.md

------------------------------------------------------------------------

## Pré-requisitos

-   Python 3.9 ou superior
-   VSCode ou outro editor Python
-   pip

------------------------------------------------------------------------

## 1. Criar ambiente virtual

No terminal do projeto:

### Windows

    python -m venv venv
    venv\Scripts\activate

### Linux / Mac

    python3 -m venv venv
    source venv/bin/activate

------------------------------------------------------------------------

## 2. Instalar dependências

Execute:

    pip install -r requirements.txt

Esse comando instalará todas as bibliotecas necessárias para o projeto.

------------------------------------------------------------------------

## 3. Gerar o dataset

O dataset é criado executando múltiplas instâncias do algoritmo memético
e registrando decisões relacionadas à aplicação de busca local.

Execute:

    python src/generate_dataset.py

O script irá:

-   gerar várias instâncias do PCV
-   executar o algoritmo memético com diferentes seeds
-   registrar informações do estado da solução
-   registrar se houve melhoria após busca local

O resultado será salvo em:

    data/decision_dataset_full.csv

------------------------------------------------------------------------

## 4. Validar o dataset

Após a geração do dataset, é recomendável verificar:

-   integridade dos dados
-   distribuição da variável alvo
-   presença de valores nulos
-   correlação entre variáveis

Validar através do "valid_decision_dataset.py"

------------------------------------------------------------------------

## 5. Treinar os modelos de Machine Learning

Após validar o dataset, execute:

    python src/train_model.py

Esse script irá:

1.  carregar o dataset
2.  separar features e target (`improved`)
3.  treinar dois modelos:

-   XGBoost
-   LightGBM

4.  avaliar o desempenho dos modelos
5.  identificar a importância das features.

------------------------------------------------------------------------

## Pipeline Experimental

    AM-PCVA-OI
          ↓
    Execução do algoritmo
          ↓
    Geração do dataset de decisões
          ↓
    Validação do dataset
          ↓
    Treinamento de modelos (XGBoost / LightGBM)
          ↓
    Modelo aprende política de decisão
          ↓
    Integração no algoritmo memético

------------------------------------------------------------------------

## Variável alvo do modelo

A variável alvo utilizada no treinamento é:

    improved

Definição:

    improved = 1 → a busca local produziu melhoria no custo do tour
    improved = 0 → a busca local não produziu melhoria

Essa variável permite treinar modelos supervisionados capazes de prever
**quando a aplicação da busca local tende a produzir ganho na solução**.

------------------------------------------------------------------------

## Observações

A implementação atual utiliza **2-opt** como busca local simplificada.
Em versões futuras, pode ser incorporada a heurística **Lin-Kernighan
(LK)** para refinamento mais sofisticado.

------------------------------------------------------------------------

## Autor

Projeto desenvolvido no contexto de **Iniciação Científica**,
investigando integração entre:

-   Metaheurísticas evolutivas
-   Problemas de otimização combinatória
-   Machine Learning aplicado a algoritmos heurísticos.
