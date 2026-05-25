# Exercício manual de BPE

## Corpus inicial
porta
portaria
porteiro

## Separação em caracteres
p o r t a
p o r t a r i a
p o r t e i r o
## Contagem inicial de pares
p o = 3
o r = 3
r t = 3
t a = 2
a r = 1
r i = 1
i a = 1
t e = 1
e i = 1
i r = 1
r o = 1

## Primeira fusão
p o => po
## Corpus após a primeira fusão
po r t a
po r t a r i a
po r t e i r o
## Segunda contagem de pares
po r = 3
r t = 3
t a = 2
a r = 1
r i = 1
i a = 1
t e = 1
e i = 1
i r = 1
r o = 1
## Segunda fusão
po r => por
## Corpus após a segunda fusão
por t a 
por t a r i a 
por t e i r o
## O que aprendi
aprendi a forma que um tokenizer identificaria e formaria os tokens que aparecen mais vezes de forma manual 