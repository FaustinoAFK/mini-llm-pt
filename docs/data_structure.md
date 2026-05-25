# Estrutura de dados

## data/raw
Será armazenado somente os arquivos brutos sem nenhuma alteração do original;
## data/processed
Será armazenado os arquivos ja processados sem nenhum lixo como rodapés, HTML, entre outro;
## data/splits
Aqui estará os arquivos finais para treinos;
## Regras
- Os dados originais nunca deve ser mudados ou editados;
- Os dados processados são derivados dos originais;
- Os splits são os arquivos finais para treinamento do modelo, validações e testes;
- Arquivos muito grandes poderá ficar fora do Git futuramente;
- Cada fonte adicionada deve respeitar os critérios do Dataset v0. 