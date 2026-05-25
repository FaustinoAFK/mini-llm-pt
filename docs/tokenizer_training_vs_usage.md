# Treinamento vs uso do tokenizer

## Treinar o tokenizer
é contar os pares e aprender fusões e assim criar um vocabulário
## Usar o tokenizer
é utilizar o aprendizado do treinamento e colocar em prática para converter palavras em tokens e os tokens em IDs posteriormente
## exemplo 
Corpus de treino:
porta
portaria
porteiro

Fusões aprendidas:
p + o → po
po + r → por

Texto novo:
portal

Aplicando fusões:
p o r t a l
→ po r t a l
→ por t a l