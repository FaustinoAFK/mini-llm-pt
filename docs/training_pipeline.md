# Pipeline de treinamento

Este documento registra o estado atual do pipeline de treinamento do projeto `mini-llm-pt`.

## Visão geral

O fluxo atual do projeto é:

```txt
data/splits/train.txt
→ CharTokenizer
→ IDs numéricos
→ exemplos x/y
→ mini-batches aleatórios
→ MiniTransformerLanguageModel
→ loss de treino
→ loss de validação
→ melhor checkpoint
→ geração de texto
```

## Dataset

O projeto usa a estrutura:

```txt
data/raw/
data/processed/
data/splits/
```

No momento, os arquivos principais de treino são:

```txt
data/splits/train.txt
data/splits/val.txt
data/splits/test.txt
```

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

O treino do Transformer agora usa mini-batches aleatórios.

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
temperature
top_k
```

`temperature` controla o nível de aleatoriedade.
`top_k` limita a escolha aos tokens mais prováveis.

## Estado em 26/05/2026

Até esta data, o projeto já possui:

```txt
pipeline de dados estruturado
CharTokenizer funcional
modelo Bigram funcional
Mini Transformer decoder-only funcional
loss de treino
loss de validação
salvamento do melhor checkpoint
mini-batches aleatórios
geração com temperature e top_k
testes automatizados com pytest
```

O próximo passo importante será adicionar mais textos ao dataset de forma controlada, mantendo o fluxo `raw → processed → splits`.
