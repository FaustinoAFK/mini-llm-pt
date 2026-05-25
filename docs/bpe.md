# BPE

## Ideia principal
A ideia principal é separar as palavra em caracteres e identificar reptições e casos que diminuiria o numeros de tokens e mesmo assim conseguiria completar uma palavra com o menor numero de passo possivel.
## Por que não usar apenas caracteres?
terião uma sequencia muit longa deixando o treinamento dos modelos mais caros. 
## Por que não usar apenas palavras?
porque iria transforma varias palavras em tokens de forma desnecessarias.
## Como o BPE aprende subpalavras?
ele aprende fusões de pares que aparecem com mais frequencia durante o treinamento do tokenizer
## Exemplo manual
casa
casas
casinha

c a s a 
c a s a s
c a s i n h a

cas 
a
s
inha 

c+a = ca por conta das repeticões no começo das palavras.
a mesma coisa acontece com o s ca+s = cas.

## Relação com o projeto
o tokenizer é nescessario para a transformação em token e posteriomente em IDs.