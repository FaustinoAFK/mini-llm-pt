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

Em 26/05/2026, o projeto passou a ter um pipeline mínimo funcional de modelo de linguagem.

Em 26/05/2026, o projeto também passou a usar um dataset maior baseado em artigos da Wikipedia em português, com fluxo `raw → processed → splits`, limpeza de ruído, verificação de qualidade e geração com prompt configurável.

### Componentes implementados até 26/05/2026

- Estrutura de dados `raw`, `processed` e `splits`;
- Manifesto de fontes do dataset;
- Manifesto de dados processados;
- Dataset com fontes da Wikipedia em português;
- Script de ingestão de artigos da Wikipedia;
- Script de processamento e limpeza dos raws;
- Script de reconstrução de splits `train`, `val` e `test`;
- Verificador de qualidade dos splits;
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
- Loss de validação estimada por mini-batches;
- Salvamento do melhor checkpoint por `val loss`;
- Treino do Transformer com `MAX_ITERS = 5000`;
- Contexto do Transformer configurado com `BLOCK_SIZE = 64`;
- Geração de texto com prompt customizável;
- Geração com `temperature` e `top_k` pela linha de comando;
- Testes automatizados com `pytest`.

### Scripts principais

```txt
scripts/train_bigram.py
scripts/generate_bigram.py
scripts/train_transformer.py
scripts/generate_transformer.py
scripts/ingest_wikipedia_raws.py
scripts/process_wikipedia_raws.py
scripts/build_splits.py
scripts/check_dataset_quality.py
```

### Documentação importante

```txt
docs/dataset_v0.md
docs/tokenizer.md
docs/bpe.md
docs/training_pipeline.md
docs/wikipedia_sources.md
```

## Uso básico

Processar dados e reconstruir splits:

```powershell
python -m scripts.process_wikipedia_raws
python -m scripts.build_splits
python -m scripts.check_dataset_quality
```

Treinar o Transformer:

```powershell
python -m scripts.train_transformer
```

Gerar texto com prompt:

```powershell
python -m scripts.generate_transformer --prompt "A inteligencia artificial"
python -m scripts.generate_transformer --prompt "Python e uma linguagem" --temperature 0.7 --top-k 5
```

## Próximo passo

Os próximos avanços técnicos prováveis são:

```txt
1. comparar o impacto do BLOCK_SIZE = 64 na val loss e na geração;
2. testar treino mais longo se a val loss continuar caindo;
3. implementar um tokenizer BPE simples para sair do nível de caractere.
```
