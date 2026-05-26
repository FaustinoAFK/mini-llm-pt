# BPE

## Ideia principal

BPE significa Byte Pair Encoding.

No contexto deste projeto, o BPE é usado como um tokenizer de subpalavras. Ele começa separando o texto em caracteres e depois aprende fusões de pares que aparecem com muita frequência.

Exemplo intuitivo:

```txt
c + a → ca
ca + s → cas
cas + a → casa
```

Com isso, palavras frequentes ou pedaços frequentes de palavras podem virar tokens maiores.

## Por que não usar apenas caracteres?

O `CharTokenizer` é simples e bom para aprender, mas gera sequências longas. Uma frase pequena vira muitos tokens.

Isso dificulta o treino porque o modelo precisa aprender palavras letra por letra.

Por isso aparecem saídas como:

```txt
solunção
esigonformações
conhecessortações
```

Essas palavras quebradas são sinal de que o modelo ainda está montando tudo caractere por caractere.

## Por que não usar apenas palavras?

Um tokenizer por palavras também tem problemas:

```txt
muitas palavras raras
muitas variações
vocabulário muito grande
problema com palavras novas
```

O BPE fica no meio termo:

```txt
maior que caractere
menor que palavra inteira
```

## Como o BPE aprende subpalavras?

O processo simplificado é:

```txt
1. começa com caracteres
2. conta pares vizinhos mais frequentes
3. junta o par mais frequente
4. repete o processo várias vezes
```

Exemplo:

```txt
banana bandana
```

Pode aprender partes como:

```txt
an
na
ban
ana
```

## Implementação atual

O projeto possui um BPE simples em:

```txt
src/tokenizer/bpe_tokenizer.py
```

Ele suporta:

```txt
treino com num_merges
encode
decode
save em JSON
load de JSON
<unk> para caracteres desconhecidos
```

Também existe um script para treinar o tokenizer:

```txt
scripts/train_bpe_tokenizer.py
```

Uso:

```powershell
python -m scripts.train_bpe_tokenizer
```

Ou com número de merges personalizado:

```powershell
python -m scripts.train_bpe_tokenizer --num-merges 1000
```

A saída padrão é:

```txt
artifacts/tokenizers/bpe.json
artifacts/tokenizers/bpe.preview.txt
```

## Relação com o projeto

O BPE ainda não substitui automaticamente o `CharTokenizer` no Transformer.

A ordem correta é:

```txt
1. implementar BPETokenizer
2. testar encode/decode/save/load
3. treinar e inspecionar o vocabulário
4. integrar no train_transformer.py
5. comparar CharTokenizer vs BPE
```

A expectativa é que o BPE reduza a quantidade de tokens e ajude o modelo a gerar palavras mais inteiras.
