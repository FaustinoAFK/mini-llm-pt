# mini-llm-pt

## Objetivo

Criar uma pequena LLM para aprendizado pratico sobre modelos de linguagem em
portugues.

O projeto e didatico: a intencao e entender dataset, tokenizacao, preparacao de
dados, arquitetura Transformer, treino, avaliacao e geracao de texto.

## O que este projeto nao e

Este projeto nao tem fins comerciais e nao pretende competir com modelos como
ChatGPT, Gemini, Claude, Llama ou similares.

Ele e uma mini-LLM de estudo, construida do zero e com escopo pequeno.

## Modulos

1. Dataset em portugues
2. Tokenizers proprios
3. Preparacao dos dados
4. Modelo Bigram
5. Modelo Transformer decoder-only
6. Treinamento
7. Avaliacao por loss
8. Geracao de texto
9. Benchmark opcional de inferencia com OpenVINO

## Estado atual

O projeto possui um pipeline funcional de modelagem de linguagem:

- Dataset em portugues com artigos da Wikipedia;
- Fluxo de dados `raw -> processed -> splits`;
- Verificacao de qualidade dos splits;
- `CharTokenizer` com suporte a `<unk>`;
- `BPETokenizer` otimizado com Hugging Face Tokenizers e `save/load`;
- Modelo `BigramLanguageModel`;
- Modelo `MiniTransformerLanguageModel` decoder-only;
- Dois caminhos de treino do Transformer:
  - por caractere, usando `CharTokenizer`;
  - por BPE, usando `BPETokenizer`;
- Atencao causal, multi-head attention, feed-forward, residual connections e
  layer normalization;
- Treino com salvamento do melhor checkpoint por `val loss`;
- Treino BPE com argumentos por CLI, early stopping e metricas JSONL;
- Geracao com prompt customizavel, `temperature` e `top_k`;
- Avaliacao qualitativa do checkpoint BPE com prompts fixos;
- Benchmark opcional PyTorch CPU vs OpenVINO;
- Testes automatizados com `pytest`;
- Configuracao de testes em `pyproject.toml`;
- Instrucoes especificas do projeto em `AGENTS.md`.

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
  evaluate_generation_bpe.py
  benchmark_openvino.py
  ingest_wikipedia_raws.py
  process_wikipedia_raws.py
  build_splits.py
  check_dataset_quality.py
```

## Instalacao

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Uso basico

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

Treinar o Transformer com BPE usando parametros customizados:

```powershell
python -m scripts.train_transformer_bpe --device auto --block-size 128 --batch-size 32 --max-iters 20000 --n-embd 256 --n-layer 4
```

Salvar metricas do treino BPE em JSONL:

```powershell
python -m scripts.train_transformer_bpe --metrics-path artifacts/runs/exp01.jsonl
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

Gerar texto com controles contra repeticao:

```powershell
python -m scripts.generate_transformer_bpe --checkpoint-path checkpoints/transformer_bpe_block128_dropout02.pt --prompt "A inteligencia artificial e " --temperature 0.7 --top-k 20 --top-p 0.9 --repetition-penalty 1.15 --no-repeat-ngram-size 3
```

Avaliar qualitativamente o checkpoint BPE com prompts fixos:

```powershell
python -m scripts.evaluate_generation_bpe
```

Comparar inferencia PyTorch CPU com OpenVINO:

```powershell
python -m scripts.benchmark_openvino --device CPU
python -m scripts.benchmark_openvino --device GPU
```

## Testes

```powershell
python -m pytest
```

## Documentacao

```txt
docs/dataset_v0.md
docs/tokenizer.md
docs/bpe.md
docs/training_pipeline.md
docs/training_experiments.md
docs/wikipedia_sources.md
```

## Proximos passos recomendados

1. Rodar experimentos comparaveis em `docs/training_experiments.md`.
2. Comparar geracoes e `val loss` entre configuracoes BPE.
3. Comparar o caminho por caractere com BPE de forma qualitativa, lembrando que
   as losses nao sao diretamente equivalentes porque as unidades de tokenizacao
   sao diferentes.
