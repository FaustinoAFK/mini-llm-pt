# mini-llm-pt

## Objetivo

Criar uma pequena LLM para aprendizado prático sobre modelos de linguagem em português.

O projeto é didático: a intenção é entender dataset, tokenização, preparação de dados,
arquitetura Transformer, treino, avaliação e geração de texto.

## O que este projeto não é

Este projeto não tem fins comerciais e não pretende competir com modelos como ChatGPT,
Gemini, Claude, Llama ou similares.

Ele é uma mini-LLM de estudo, construída do zero e com escopo pequeno.

## Módulos

1. Dataset em português
2. Tokenizers próprios
3. Preparação dos dados
4. Modelo Bigram
5. Modelo Transformer decoder-only
6. Treinamento
7. Avaliação por loss
8. Geração de texto

## Estado atual

O projeto possui um pipeline funcional de modelagem de linguagem:

- Dataset em português com artigos da Wikipedia;
- Fluxo de dados `raw -> processed -> splits`;
- Verificação de qualidade dos splits;
- `CharTokenizer` com suporte a `<unk>`;
- `BPETokenizer` simples;
- Criação de exemplos `x/y` para previsão do próximo token;
- Mini-batches aleatórios;
- `BigramLanguageModel`;
- `MiniTransformerLanguageModel` decoder-only;
- Atenção causal;
- Multi-head attention;
- Feed-forward;
- Residual connections;
- Layer normalization;
- Treino com salvamento do melhor checkpoint por `val loss`;
- Geração com prompt customizável;
- Controle de geração por `temperature` e `top_k`;
- Testes automatizados com `pytest`.

## Estrutura principal

```txt
src/
  batching.py
  data_loader.py
  evaluation.py
  training_data.py
  models/
    bigram.py
    transformer.py
  tokenizer/
    char_tokenizer.py
    bpe_tokenizer.py

scripts/
  train_bigram.py
  generate_bigram.py
  train_bpe_tokenizer.py
  train_transformer.py
  train_transformer_bpe.py
  generate_transformer.py
  generate_transformer_bpe.py
  ingest_wikipedia_raws.py
  process_wikipedia_raws.py
  build_splits.py
  check_dataset_quality.py
```

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Uso básico

Processar dados e reconstruir splits:

```powershell
python -m scripts.process_wikipedia_raws
python -m scripts.build_splits
python -m scripts.check_dataset_quality
```

Treinar o tokenizer BPE:

```powershell
python -m scripts.train_bpe_tokenizer
```

Treinar o Transformer por caractere:

```powershell
python -m scripts.train_transformer
```

Treinar o Transformer com BPE:

```powershell
python -m scripts.train_transformer_bpe
```

Gerar texto com o Transformer por caractere:

```powershell
python -m scripts.generate_transformer --prompt "A inteligencia artificial"
python -m scripts.generate_transformer --prompt "Python e uma linguagem" --temperature 0.7 --top-k 5
```

Gerar texto com o Transformer BPE:

```powershell
python -m scripts.generate_transformer_bpe --prompt "A inteligencia artificial "
```

## Testes

```powershell
python -m pytest -q
```

## Documentação

```txt
docs/dataset_v0.md
docs/tokenizer.md
docs/bpe.md
docs/training_pipeline.md
docs/wikipedia_sources.md
```

## Próximos passos recomendados

1. Expor hiperparâmetros dos scripts de treino via argumentos de linha de comando.
2. Comparar resultados entre tokenizer por caractere e BPE.
3. Registrar métricas de treino em arquivo para acompanhar evolução.
4. Experimentar `BLOCK_SIZE`, `N_EMBD`, `N_LAYER`, `N_HEAD` e tamanho do dataset.
5. Melhorar a avaliação qualitativa da geração com prompts fixos.
