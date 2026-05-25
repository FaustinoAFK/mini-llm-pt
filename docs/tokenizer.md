# Tokenizer

## Objetivo
transformar as palavras em tokens e depois para IDs para que possa ser ultilizado pelo modelo de alguma forma.
## Por que uma LLM precisa de tokenizer?
Uma LLM nescessita de um tokenizer porque ela sozinha não compreende texto cru, nisso entra o tokenizer para tra sformar as palavras em tokens e depois em IDs numericos. e esses IDs são tranformados em embeddings
## Texto vs tokens
textos é a forma da escrita que humanos entenden e compreendem e tokens é a forma que a LLM entende e comprende da forma dela
Texto:
"Eu gosto de IA."

Tokens possíveis:
["Eu", " gosto", " de", " IA", "."]

IDs possíveis:
[10, 45, 82, 300, 7]

## Tipos de tokenização
Podem ser de varias forma desde caracteres, palavras e subpalavras.
## Tokenizer escolhido para o projeto
O projeto pretende usar um tokenizer por subpalavras, como BPE, por ser mais próximo do usado em modelos modernos.
## Estado atual