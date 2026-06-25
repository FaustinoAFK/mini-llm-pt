# Google Colab

Este projeto pode ser treinado no Google Colab com o notebook:

- `notebooks/mini_llm_pt_colab.ipynb`

## O que o notebook faz

- monta o Google Drive;
- localiza ou clona o repositorio;
- instala dependencias;
- opcionalmente reconstrui dataset e tokenizer;
- treina o Transformer BPE com checkpoint e metricas nomeados por experimento;
- salva relatorio de geracao em `artifacts/evaluations/`.

## Como usar

1. Abra o notebook no Colab.
2. Troque o runtime para GPU.
3. Ajuste a celula de configuracao:
   - `REPO_URL` se o projeto ainda nao estiver no Drive;
   - `EXP_NAME`;
   - flags como `RETRAIN_TOKENIZER` e `REBUILD_DATASET`;
   - hiperparametros do treino.
4. Rode as celulas em ordem.

## Saidas principais

- checkpoint: `checkpoints/<EXP_NAME>.pt`
- metricas: `artifacts/runs/<EXP_NAME>.jsonl`
- avaliacao: `artifacts/evaluations/<EXP_NAME>_generation.txt`

## Observacoes

- Se o runtime cair, reabra o notebook, monte o Drive e preencha `RESUME_FROM`
  com o checkpoint salvo do experimento.
- O notebook foi pensado para o pipeline BPE atual do projeto.
