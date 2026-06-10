# Mapa mental de arquivos e modulos

Este documento mapeia a pasta `mini-llm-pt` com foco em funcao, relacoes entre
modulos e fluxo do pipeline. A ideia e servir como guia de navegacao rapida.

## Visao mental do projeto

```txt
mini-llm-pt
|
+-- README e configuracao
|   +-- README.md
|   +-- AGENTS.md
|   +-- pyproject.toml
|   +-- requirements.txt
|
+-- Codigo reutilizavel (src/)
|   +-- tokenizacao
|   |   +-- char_tokenizer.py
|   |   `-- bpe_tokenizer.py
|   +-- modelos
|   |   +-- bigram.py
|   |   `-- transformer.py
|   `-- utilitarios de treino e avaliacao
|       +-- batching.py
|       +-- data_loader.py
|       +-- evaluation.py
|       `-- training_data.py
|
+-- Scripts executaveis (scripts/)
|   +-- pipeline de dados Wikipedia
|   +-- treino de tokenizer
|   +-- treino de modelos
|   +-- geracao e avaliacao
|   `-- benchmark e manutencao
|
+-- Testes (tests/)
|   +-- testes de utilitarios
|   +-- testes de tokenizers
|   +-- testes de modelos
|   `-- testes de scripts criticos
|
+-- Documentacao (docs/)
|   +-- dataset
|   +-- tokenizacao
|   +-- pipeline de treino
|   `-- este mapa mental
|
`-- Dados e artefatos
    +-- data/raw
    +-- data/processed
    +-- data/splits
    +-- artifacts/
    `-- checkpoints/
```

## Raiz do projeto

- `README.md`
  - Porta de entrada do projeto.
  - Explica objetivo, pipeline, comandos e estado atual.
  - Resume os modulos principais e o fluxo `raw -> processed -> splits -> treino -> geracao`.

- `AGENTS.md`
  - Regras especificas do projeto.
  - Define objetivo didatico, tecnologias, comandos e cuidados com dados e checkpoints.

- `pyproject.toml`
  - Configuracao do `pytest`.
  - Define `pythonpath`, pasta de testes e `addopts = "-q"`.

- `requirements.txt`
  - Dependencias Python do projeto.
  - Base para instalar PyTorch, pytest, tokenizers e suporte opcional a benchmark.

- `.gitignore`
  - Regras de exclusao do Git.

## Codigo reutilizavel (`src/`)

### Visao por modulo

```txt
src/
|
+-- batching.py
+-- data_loader.py
+-- evaluation.py
+-- training_data.py
|
+-- tokenizer/
|   +-- __init__.py
|   +-- char_tokenizer.py
|   `-- bpe_tokenizer.py
|
`-- models/
    +-- __init__.py
    +-- bigram.py
    `-- transformer.py
```

### Arquivo a arquivo

- `src/batching.py`
  - Funcao `get_batch(x, y, batch_size)`.
  - Sorteia mini-batches aleatorios a partir de tensores ja preparados.
  - E usado no treino do Transformer por caractere e na avaliacao por batches.

- `src/data_loader.py`
  - `read_text_file(path)`: leitura UTF-8 simples.
  - `encode_text_file(path, tokenizer)`: leitura seguida de tokenizacao.
  - Centraliza IO minimo de texto para os scripts.

- `src/evaluation.py`
  - `estimate_loss(model, x, y)`: loss sem atualizar pesos.
  - `estimate_loss_over_batches(...)`: media de loss em varios mini-batches.
  - Cuida de alternar `train/eval` sem deixar o modelo em estado errado.

- `src/training_data.py`
  - `create_training_example(ids, block_size)`: um par `(x, y)`.
  - `create_training_examples(ids, block_size)`: janela deslizante sobre IDs.
  - `train_val_split_ids(ids, train_ratio)`: split simples em memoria.
  - E a base do pipeline por caractere.

### Submodulo `src/tokenizer/`

- `src/tokenizer/__init__.py`
  - Marca a pasta como pacote Python.

- `src/tokenizer/char_tokenizer.py`
  - Implementa `CharTokenizer`.
  - Descobre vocabulario por caractere a partir do texto de treino.
  - Usa `<unk>` para caracteres fora do vocabulario conhecido.
  - Caminho mais simples e didatico de tokenizacao do projeto.

- `src/tokenizer/bpe_tokenizer.py`
  - Implementa `BPETokenizer`.
  - Usa a biblioteca `tokenizers` com modelo BPE e `ByteLevel`.
  - Faz `train`, `encode`, `decode`, `save` e `load`.
  - Mantem compatibilidade com formato legado via `_LegacyTokenizerAdapter`.
  - E o tokenizer mais avancado e o mais importante no pipeline atual.

### Submodulo `src/models/`

- `src/models/__init__.py`
  - Marca a pasta como pacote Python.

- `src/models/bigram.py`
  - Implementa `BigramLanguageModel`.
  - Modelo minimo: embedding de tamanho `vocab_size x vocab_size`.
  - Serve para ensinar o ciclo completo: logits, loss e geracao autoregressiva.

- `src/models/transformer.py`
  - Implementa a espinha dorsal do modelo decoder-only:
    - `CausalSelfAttentionHead`
    - `MultiHeadCausalSelfAttention`
    - `FeedForward`
    - `TransformerBlock`
    - `MiniTransformerLanguageModel`
  - Tambem concentra a logica de geracao com:
    - `temperature`
    - `top_k`
    - `top_p`
    - `repetition_penalty`
    - `no_repeat_ngram_size`
  - E o modulo central de modelagem de linguagem.

## Scripts executaveis (`scripts/`)

### Visao por grupo

```txt
scripts/
|
+-- dados Wikipedia
|   +-- ingest_wikipedia_raws.py
|   +-- process_wikipedia_raws.py
|   +-- consolidate_wikipedia_raws.py
|   +-- build_splits.py
|   `-- check_dataset_quality.py
|
+-- tokenizacao
|   +-- train_bpe_tokenizer.py
|   `-- analyze_bpe_tokenizer.py
|
+-- treino
|   +-- train_bigram.py
|   +-- train_transformer.py
|   `-- train_transformer_bpe.py
|
+-- inferencia e avaliacao
|   +-- generate_bigram.py
|   +-- generate_transformer.py
|   +-- generate_transformer_bpe.py
|   `-- evaluate_generation_bpe.py
|
`-- manutencao e benchmark
    +-- benchmark_openvino.py
    `-- fix_mojibake.py
```

### Arquivo a arquivo

- `scripts/ingest_wikipedia_raws.py`
  - Faz ingestao de textos da Wikipedia a partir de links.
  - Extrai URLs de markdown, busca JSON/API da Wikipedia, gera nomes de arquivo e manifesto.
  - Alimenta `data/raw/wikipedia/`.

- `scripts/process_wikipedia_raws.py`
  - Limpa os textos brutos da Wikipedia.
  - Remove secoes, ruido simbolico, formulas, excesso de numeros e marcas indesejadas.
  - Gera `data/processed/wikipedia/` e um manifesto com estatisticas.
  - E o filtro principal de qualidade textual antes dos splits.

- `scripts/consolidate_wikipedia_raws.py`
  - Consolida arquivos de retry/fontes adicionais para a area raw principal.
  - Ajuda a reorganizar aquisicao de dados sem misturar tudo manualmente.

- `scripts/build_splits.py`
  - Le textos processados, embaralha conforme seed e cria `train.txt`, `val.txt`, `test.txt`.
  - Gera manifesto dos splits com rastreabilidade.
  - Formaliza a fronteira entre corpus processado e dataset de treino.

- `scripts/check_dataset_quality.py`
  - Mede tamanho, linhas, palavras e outros indicadores basicos dos splits.
  - Funciona como cheque rapido de sanidade do dataset.

- `scripts/train_bpe_tokenizer.py`
  - Treina o `BPETokenizer` a partir de `data/splits/train.txt`.
  - Salva em `artifacts/tokenizers/bpe.json`.

- `scripts/analyze_bpe_tokenizer.py`
  - Analisa o tokenizer BPE treinado.
  - Resume tipos de token, comprimentos e exemplos.
  - Ajuda a inspecionar se o vocabulario aprendido faz sentido.

- `scripts/train_bigram.py`
  - Pipeline de treino mais simples.
  - Usa `CharTokenizer`, `create_training_examples` e `BigramLanguageModel`.
  - Salva checkpoint em `checkpoints/bigram.pt`.

- `scripts/generate_bigram.py`
  - Reabre checkpoint do Bigram e gera texto.
  - Serve como demonstracao da primeira etapa de modelagem.

- `scripts/train_transformer.py`
  - Treino do Transformer por caractere.
  - Usa `CharTokenizer`, batching aleatorio e avaliacao por `val loss`.
  - Salva o melhor checkpoint em `checkpoints/transformer.pt`.

- `scripts/generate_transformer.py`
  - Gera texto a partir do checkpoint do Transformer por caractere.
  - Reconstrui tokenizer e modelo a partir do checkpoint salvo.

- `scripts/train_transformer_bpe.py`
  - Script mais completo do projeto hoje.
  - Treina Transformer com `BPETokenizer`.
  - Traz CLI rica com:
    - escolha de device;
    - `warmup`;
    - cosine decay;
    - gradient clipping;
    - gradient accumulation;
    - early stopping;
    - `resume_from`;
    - metricas JSONL;
    - checkpoint com estado de treino e RNG.
  - E o centro do pipeline atual de experimentacao.

- `scripts/generate_transformer_bpe.py`
  - Inference principal do caminho BPE.
  - Carrega checkpoint e tokenizer associados.
  - Exponibiliza controles de geracao mais avancados.

- `scripts/evaluate_generation_bpe.py`
  - Avalia qualitativamente o checkpoint BPE com prompts fixos.
  - Gera relatorio textual/estruturado das respostas.
  - Fecha o ciclo de treino + observacao de qualidade.

- `scripts/benchmark_openvino.py`
  - Compara inferencia do modelo em PyTorch CPU e OpenVINO.
  - Usa um wrapper `LogitsOnlyModel` para benchmark.
  - E opcional e mais voltado a desempenho.

- `scripts/fix_mojibake.py`
  - Ferramenta de manutencao para corrigir mojibake em arquivos de texto.
  - Mede marcadores de corrupcao, decide se a reparacao vale a pena e pode rodar em dry-run.
  - E util para higiene do dataset e documentacao em portugues.

## Testes (`tests/`)

### Ideia geral

Os testes espelham o codigo do projeto. Em vez de testar tudo via script de ponta
a ponta, a suite cobre utilitarios, tokenizers, modelos e o script BPE mais
importante.

### Arquivo a arquivo

- `tests/__init__.py`
  - Marca a pasta como pacote.

- `tests/test_batching.py`
  - Garante que `get_batch` devolve shapes corretos e rejeita entradas invalidas.

- `tests/test_data_loader.py`
  - Verifica leitura de arquivo e encode via tokenizer.

- `tests/test_evaluation.py`
  - Testa calculo de loss e preservacao correta do modo `train/eval`.

- `tests/test_training_data.py`
  - Cobre janela deslizante, validacoes e split simples de IDs.

- `tests/test_char_tokenizer.py`
  - Garante roundtrip, uso de `<unk>` e tamanho do vocabulario.

- `tests/test_bpe_tokenizer.py`
  - Cobre treino, roundtrip, caracteres nao vistos, `save/load` e validacoes.

- `tests/test_bigram.py`
  - Verifica `forward`, loss e geracao do `BigramLanguageModel`.

- `tests/test_transformer.py`
  - Cobre `forward`, loss, geracao e validacoes dos controles de sampling.
  - Tambem testa regras de repeticao e limite de `block_size`.

- `tests/test_fix_mojibake.py`
  - Garante reparo de casos comuns de texto corrompido e comportamento de dry-run/apply.

- `tests/test_train_transformer_bpe.py`
  - Cobre validacao de argumentos, scheduler e fluxo resumido de checkpoint/resume.
  - E o teste mais proximo de uma verificacao funcional do pipeline BPE.

## Documentacao (`docs/`)

### Arquivo a arquivo

- `docs/dataset_v0.md`
  - Define criterios e escopo inicial do dataset.

- `docs/data_structure.md`
  - Explica a funcao de `data/raw`, `data/processed` e `data/splits`.

- `docs/splits_structure.md`
  - Resume a estrutura esperada dos arquivos de split.

- `docs/tokenizer.md`
  - Documentacao conceitual sobre tokenizacao no projeto.

- `docs/tokenizer_training_vs_usage.md`
  - Separa mentalmente o ato de treinar um tokenizer do ato de usa-lo.

- `docs/bpe.md`
  - Explica a abordagem BPE no contexto do projeto.

- `docs/bpe_manual_exercise.md`
  - Exercicio manual para entender BPE passo a passo.

- `docs/training_pipeline.md`
  - Explica o pipeline de treino do projeto.

- `docs/training_experiments.md`
  - Registra experimentos e comparacoes de treino.

- `docs/wikipedia_sources.md`
  - Lista e documenta fontes da Wikipedia usadas no corpus.

- `docs/wikipedia_retry_sources.md`
  - Registro de fontes/retentativas complementares.

- `docs/wikipedia_retry_sources_v2.md`
  - Segunda iteracao do registro de retentativas/fontes.

- `docs/mapa_mental_arquivos.md`
  - Este documento.

- `docs/docs/`
  - Subpasta atualmente vazia.
  - Vale revisar se foi criada por engano ou reservada para futura documentacao.

## Dados e artefatos

### `data/`

```txt
data/
|
+-- raw/
|   `-- wikipedia/
|
+-- processed/
|   `-- wikipedia/
|
`-- splits/
    +-- train.txt
    +-- val.txt
    +-- test.txt
    `-- MANIFEST.md
```

- `data/raw/wikipedia/`
  - Guarda textos brutos capturados da Wikipedia.
  - E materia-prima; nao deve ser tratada como dataset final.

- `data/processed/wikipedia/`
  - Guarda textos limpos apos o processamento.
  - Costuma conter muitos arquivos de artigo; o mapeamento util aqui e por etapa, nao por arquivo individual.

- `data/splits/train.txt`
  - Base principal de treino dos modelos e do tokenizer BPE.

- `data/splits/val.txt`
  - Base de validacao para acompanhar `val loss`.

- `data/splits/test.txt`
  - Base reservada para avaliacao separada de treino/validacao.

- `data/splits/MANIFEST.md`
  - Documento de rastreabilidade dos splits gerados.

### `artifacts/`

- `artifacts/tokenizers/`
  - Guarda tokenizers treinados e arquivos auxiliares de inspecao.

- `artifacts/runs/`
  - Guarda metricas e saidas de experimentos, especialmente JSONL do treino BPE.

### `checkpoints/`

- Guarda pesos salvos dos modelos.
- Em especial:
  - Bigram;
  - Transformer por caractere;
  - Transformer BPE.

## Fluxo mental recomendado para navegar no projeto

1. Ler `README.md`.
2. Entender `docs/data_structure.md` e `docs/training_pipeline.md`.
3. Ver `src/tokenizer/` e `src/models/`.
4. Estudar `scripts/train_transformer.py` e `scripts/train_transformer_bpe.py`.
5. Fechar com `tests/` para ver o comportamento esperado.

## Ponto de orientacao rapido

Se a duvida for "onde mexer?", use este atalho mental:

- problema de leitura ou arquivo: `src/data_loader.py`
- problema de exemplos ou janela: `src/training_data.py`
- problema de batch ou loss: `src/batching.py` e `src/evaluation.py`
- problema de tokenizacao simples: `src/tokenizer/char_tokenizer.py`
- problema de tokenizacao BPE: `src/tokenizer/bpe_tokenizer.py`
- problema de arquitetura do modelo: `src/models/bigram.py` ou `src/models/transformer.py`
- problema de dataset Wikipedia: `scripts/ingest_wikipedia_raws.py`, `scripts/process_wikipedia_raws.py`, `scripts/build_splits.py`
- problema de treino moderno: `scripts/train_transformer_bpe.py`
- problema de geracao: `scripts/generate_transformer.py` ou `scripts/generate_transformer_bpe.py`
- problema de regressao: procurar o teste espelho em `tests/`
