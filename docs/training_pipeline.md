# Pipeline de treinamento

Este documento registra o estado atual do pipeline de treinamento do projeto `mini-llm-pt`.

## Visão geral

O fluxo atual do projeto é:

```txt
data/raw/wikipedia/
→ data/processed/wikipedia/
→ data/splits/train.txt
→ CharTokenizer
→ IDs numéricos
→ exemplos x/y
→ mini-batches aleatórios
→ MiniTransformerLanguageModel
→ loss de treino
→ loss de validação
→ melhor checkpoint
→ geração de texto com prompt
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

Os arquivos principais de treino são:

```txt
data/splits/train.txt
data/splits/val.txt
data/splits/test.txt
```

## Ingestão de dados

O script abaixo baixa artigos listados em arquivos markdown de fontes:

```txt
scripts/ingest_wikipedia_raws.py
```

Ele lê links da Wikipedia, baixa o texto puro via API pública da Wikipedia, salva arquivos em `data/raw/wikipedia/` e cria manifesto de fontes.

## Processamento dos raws

O script abaixo limpa os textos raw:

```txt
scripts/process_wikipedia_raws.py
```

A limpeza atual aplica:

```txt
normalização Unicode NFKC
remoção de títulos de seção
remoção de linhas com muito ruído simbólico
remoção de linhas com termos LaTeX/MediaWiki
remoção de linhas dominadas por números
filtragem de caracteres incomuns
normalização de espaços
```

A saída fica em:

```txt
data/processed/wikipedia/
```

## Splits

O script abaixo reconstrói os splits:

```txt
scripts/build_splits.py
```

Ele junta os textos processados e gera:

```txt
data/splits/train.txt
data/splits/val.txt
data/splits/test.txt
```

A proporção padrão é:

```txt
train: 80%
val: 10%
test: 10%
```

## Verificação de qualidade

O script abaixo verifica caracteres suspeitos nos splits:

```txt
scripts/check_dataset_quality.py
```

Ele ajuda a detectar ruído antes do treino.

## Tokenizer

O tokenizer atual é o `CharTokenizer`.

Ele transforma texto em caracteres, converte cada caractere para um ID numérico e também consegue fazer o caminho inverso.

Também existe suporte ao token especial:

```txt
<unk>
```

Esse token é usado quando aparece um caractere desconhecido.

## Exemplos de treino

O projeto cria pares `x` e `y` para previsão do próximo token.

Exemplo:

```txt
ids = [1, 2, 3, 4]

x = [1, 2, 3]
y = [2, 3, 4]
```

O `x` é o contexto recebido pelo modelo.
O `y` é o alvo deslocado uma posição à frente.

## Mini-batches

O treino do Transformer usa mini-batches aleatórios.

Isso significa que, a cada step, o modelo vê apenas uma amostra aleatória dos exemplos de treino, em vez de usar todos os exemplos ao mesmo tempo.

Esse comportamento deixa o treino mais parecido com projetos reais de machine learning.

## Modelo Bigram

O projeto possui um `BigramLanguageModel`.

Ele foi usado como primeiro modelo de linguagem para validar o ciclo básico:

```txt
IDs → logits → loss → treino → checkpoint → geração
```

O Bigram olha apenas para o token atual para tentar prever o próximo token.

## Mini Transformer decoder-only

O projeto possui um `MiniTransformerLanguageModel`.

Ele contém:

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

Esse é o primeiro modelo do projeto com estrutura parecida com uma LLM moderna, em escala pequena.

## Configuração atual de treino

A configuração atual do Transformer é:

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

O `BLOCK_SIZE = 64` permite que o modelo veja mais contexto do que a configuração anterior de 32 caracteres.

## Avaliação

O treino do Transformer calcula:

```txt
batch loss
train loss
val loss
best val loss
```

A `train loss` mede o desempenho no texto de treino.
A `val loss` mede o desempenho em texto separado de validação.

Para evitar uso excessivo de memória, `train loss` e `val loss` são estimadas por mini-batches, não pelo dataset inteiro de uma vez.

Quando a `train loss` cai, mas a `val loss` começa a subir, isso indica overfitting.

## Checkpoint

O script `scripts/train_transformer.py` salva o melhor checkpoint com base na menor `val loss`.

O checkpoint fica em:

```txt
checkpoints/transformer.pt
```

Esse arquivo não deve ser enviado para o GitHub.

## Geração

O script `scripts/generate_transformer.py` carrega o checkpoint treinado e gera texto.

A geração suporta:

```txt
prompt
temperature
top_k
max_new_tokens
```

Exemplos:

```powershell
python -m scripts.generate_transformer --prompt "A inteligencia artificial"
python -m scripts.generate_transformer --prompt "Python e uma linguagem" --temperature 0.7 --top-k 5
```

`temperature` controla o nível de aleatoriedade.
`top_k` limita a escolha aos tokens mais prováveis.

## Estado em 26/05/2026

Até esta data, o projeto já possui:

```txt
pipeline de dados estruturado
ingestão de artigos da Wikipedia
processamento e limpeza dos raws
reconstrução de splits
verificação de qualidade do dataset
CharTokenizer funcional
modelo Bigram funcional
Mini Transformer decoder-only funcional
loss de treino
loss de validação estimada por mini-batches
salvamento do melhor checkpoint
mini-batches aleatórios
geração com prompt, temperature e top_k
testes automatizados com pytest
```

Os próximos passos prováveis são comparar o impacto de `BLOCK_SIZE = 64`, testar treino mais longo se a `val loss` continuar caindo e implementar um tokenizer BPE simples.
