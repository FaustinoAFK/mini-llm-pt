# mini-llm-pt

## Objetivo

Criar uma pequena LLM para fins de aprendizado, estudo e conhecimento prático da área de modelos de linguagem.

## Escopo da versão 0

- Dataset inicial em português;
- Tokenizer próprio;
- Modelo Transformer decoder-only próprio;
- Treinamento para prever o próximo token;
- Avaliação por loss;
- Geração de texto a partir de checkpoints treinados.

## O que este projeto não é

Este projeto não tem fins comerciais e não pretende competir com modelos como ChatGPT, Gemini, Claude ou Llama.

O objetivo principal é aprendizado técnico e construção gradual de uma mini-LLM do zero.

## Trilha de aprendizado

## Módulos

1. Dataset
2. Tokenizer
3. Preparação dos dados
4. Embeddings
5. Transformer
6. Treinamento
7. Avaliação
8. Geração de texto

## Estado atual

Projeto iniciado no dia 25/05/2026.

Em 26/05/2026, o projeto já possui um pipeline mínimo funcional de modelo de linguagem.

### Componentes implementados até 26/05/2026

- Estrutura de dados `raw`, `processed` e `splits`;
- Manifesto de fontes do dataset;
- Manifesto de dados processados;
- `CharTokenizer` funcional com suporte a `<unk>`;
- Conversão de texto para IDs numéricos;
- Criação de exemplos `x/y` para previsão do próximo token;
- Mini-batches aleatórios;
- `BigramLanguageModel`;
- `MiniTransformerLanguageModel` decoder-only;
- Causal self-attention;
- Multi-head attention;
- Feed-forward;
- Residual connections;
- Layer normalization;
- Loss de treino;
- Loss de validação;
- Salvamento do melhor checkpoint por `val loss`;
- Geração de texto com `temperature` e `top_k`;
- Testes automatizados com `pytest`.

### Scripts principais

```txt
scripts/train_bigram.py
scripts/generate_bigram.py
scripts/train_transformer.py
scripts/generate_transformer.py
```

### Documentação importante

```txt
docs/dataset_v0.md
docs/tokenizer.md
docs/bpe.md
docs/training_pipeline.md
```

## Próximo passo

O próximo passo importante será aumentar o dataset de forma controlada, mantendo o fluxo:

```txt
data/raw → data/processed → data/splits
```

Depois disso, o modelo poderá ser treinado com mais exemplos e avaliado por `train loss` e `val loss`.
