# Pipeline de treinamento

Este documento registra o estado atual do pipeline de treinamento do projeto
`mini-llm-pt`.

## Visao geral

O fluxo de dados principal e:

```txt
data/raw/wikipedia/
-> data/processed/wikipedia/
-> data/splits/train.txt
-> data/splits/val.txt
-> data/splits/test.txt
```

A partir dos splits, existem dois caminhos de modelagem com Transformer:

```txt
train.txt -> CharTokenizer -> train_transformer.py -> checkpoints/transformer.pt
train.txt -> BPETokenizer -> train_transformer_bpe.py -> checkpoints/transformer_bpe.pt
```

O caminho BPE tambem pode salvar metricas em JSONL e gerar um relatorio
qualitativo com prompts fixos:

```txt
train_transformer_bpe.py
-> artifacts/runs/*.jsonl
-> checkpoints/*.pt
-> evaluate_generation_bpe.py
-> artifacts/evaluations/*.txt
```

## Dataset

O projeto usa a estrutura:

```txt
data/raw/
data/processed/
data/splits/
```

Os textos da Wikipedia ficam em:

```txt
data/raw/wikipedia/
data/processed/wikipedia/
```

Os arquivos finais de treino sao:

```txt
data/splits/train.txt
data/splits/val.txt
data/splits/test.txt
```

## Ingestao e processamento

O script `scripts/ingest_wikipedia_raws.py` baixa artigos listados nos arquivos
de fontes da Wikipedia, salva textos puros em `data/raw/wikipedia/` e atualiza o
manifesto de fontes.

O script `scripts/process_wikipedia_raws.py` limpa os textos raw e salva a saida
em `data/processed/wikipedia/`. A limpeza atual aplica normalizacao Unicode,
remocao de titulos de secao, filtragem de linhas ruidosas, remocao de trechos
com padroes de LaTeX/MediaWiki e normalizacao de espacos.

O script `scripts/build_splits.py` cria `train.txt`, `val.txt` e `test.txt` a
partir dos textos processados. A proporcao padrao e:

```txt
train: 80%
val: 10%
test: 10%
```

O script `scripts/check_dataset_quality.py` verifica caracteres suspeitos nos
splits antes do treino.

## Tokenizers

O projeto possui dois tokenizers proprios:

- `CharTokenizer`: transforma cada caractere em um ID e usa `<unk>` para
  caracteres desconhecidos.
- `BPETokenizer`: usa Hugging Face Tokenizers com BPE byte-level, suporta
  `<unk>` e persiste vocabulario/merges em JSON.

O tokenizer BPE e treinado com:

```powershell
python -m scripts.train_bpe_tokenizer
```

Por padrao, ele le `data/splits/train.txt` e salva:

```txt
artifacts/tokenizers/bpe.json
artifacts/tokenizers/bpe.preview.txt
```

## Exemplos e mini-batches

O treino cria pares `x` e `y` para previsao do proximo token:

```txt
ids = [1, 2, 3, 4]

x = [1, 2, 3]
y = [2, 3, 4]
```

No caminho por caractere, `train_transformer.py` materializa janelas deslizantes
com `create_training_examples` e sorteia mini-batches com `get_batch`.

No caminho BPE, `train_transformer_bpe.py` trabalha diretamente sobre o tensor
de IDs tokenizados e sorteia posicoes iniciais a cada batch. Esse fluxo evita
materializar todas as janelas e facilita treinos com textos maiores.

## Modelos

O projeto possui:

- `BigramLanguageModel`, usado para validar o ciclo basico de linguagem;
- `MiniTransformerLanguageModel`, um Transformer decoder-only pequeno.

O Transformer contem:

```txt
token embeddings
position embeddings
causal self-attention
multi-head attention
feed-forward
residual connections
layer normalization
lm head
```

## Treino por caractere

O script `scripts/train_transformer.py` treina o Transformer com `CharTokenizer`.
Ele usa constantes internas para configuracao e salva o melhor checkpoint em:

```txt
checkpoints/transformer.pt
```

Configuracao atual:

```txt
BLOCK_SIZE = 64
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EVAL_NUM_BATCHES = 20
MAX_ITERS = 5000
EVAL_INTERVAL = 500
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
DROPOUT = 0.1
```

## Treino com BPE

O script `scripts/train_transformer_bpe.py` treina o Transformer com
`BPETokenizer`. Ele aceita parametros por CLI para caminhos, hiperparametros,
device, early stopping, seed, metricas, scheduler, gradient clipping,
gradient accumulation e retomada de checkpoint.

Comando basico:

```powershell
python -m scripts.train_transformer_bpe
```

Exemplo com parametros customizados:

```powershell
python -m scripts.train_transformer_bpe --device auto --block-size 128 --batch-size 32 --max-iters 20000 --n-embd 256 --n-layer 4
```

Configuracao padrao atual:

```txt
BLOCK_SIZE = 64
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EVAL_NUM_BATCHES = 20
MAX_ITERS = 10000
EVAL_INTERVAL = 1000
PATIENCE = 2
N_EMBD = 128
N_HEAD = 4
N_LAYER = 3
DROPOUT = 0.1
DEVICE = auto
```

Por padrao, o treino BPE usa:

```txt
artifacts/tokenizers/bpe.json
checkpoints/transformer_bpe.pt
artifacts/runs/transformer_bpe_metrics.jsonl
```

Cada linha do JSONL registra eventos de inicio, avaliacao e fim do treino.
Os eventos de avaliacao tambem registram learning rate real, norma do gradiente,
tokens por step, tokens por segundo e tempo decorrido.

Controles adicionais uteis:

```powershell
python -m scripts.train_transformer_bpe --grad-clip 1.0 --warmup-iters 100 --min-learning-rate 1e-4
python -m scripts.train_transformer_bpe --disable-lr-schedule
python -m scripts.train_transformer_bpe --resume-from checkpoints/transformer_bpe.pt
python -m scripts.train_transformer_bpe --gradient-accumulation-steps 2
```

## Avaliacao e geracao

Durante o treino, os scripts acompanham `batch loss`, `train loss`, `val loss` e
`best val loss`. A `val loss` e usada para escolher o melhor checkpoint.

Geracao por caractere:

```powershell
python -m scripts.generate_transformer --prompt "A inteligencia artificial"
```

Geracao BPE:

```powershell
python -m scripts.generate_transformer_bpe --prompt "A inteligencia artificial "
```

Avaliacao qualitativa BPE com prompts fixos:

```powershell
python -m scripts.evaluate_generation_bpe
```

O relatorio padrao fica em:

```txt
artifacts/evaluations/transformer_bpe_generation.txt
```

## Benchmark OpenVINO

O script `scripts.benchmark_openvino` compara o forward do modelo BPE em PyTorch
CPU com OpenVINO:

```powershell
python -m scripts.benchmark_openvino --device CPU
```

O uso de GPU depende dos dispositivos disponiveis no OpenVINO local:

```powershell
python -m scripts.benchmark_openvino --device GPU
```

## Artefatos

Checkpoints, metricas e relatorios de avaliacao nao devem ser versionados. Os
caminhos atuais ja estao cobertos por `.gitignore`:

```txt
checkpoints/
artifacts/runs/
artifacts/evaluations/
*.pt
```

O tokenizer BPE em `artifacts/tokenizers/bpe.json` e mantido no repositorio como
artefato pequeno e util para reproduzir geracao/treino BPE.

## Experimentos

A matriz inicial de experimentos esta em:

```txt
docs/training_experiments.md
```

Ela prioriza comparacoes simples entre baseline BPE, contexto maior, modelo
maior e uma comparacao conservadora entre caractere e BPE.
