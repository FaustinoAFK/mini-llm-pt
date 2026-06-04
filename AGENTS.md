# Instrucoes para o projeto mini-llm-pt

## Objetivo

Projeto didatico para construir e estudar uma mini-LLM em portugues, cobrindo dataset, tokenizacao, preparacao de dados, modelos Bigram e Transformer decoder-only, treino, avaliacao e geracao.

## Tecnologias

- Python
- PyTorch
- pytest
- OpenVINO para benchmark opcional de inferencia

## Estilo de codigo

- Prefira codigo simples, explicito e didatico.
- Mantenha nomes e mensagens em portugues quando o arquivo ja seguir esse padrao.
- Evite abstracoes novas sem necessidade clara.
- Preserve a estrutura atual: `src/` para codigo reutilizavel, `scripts/` para comandos executaveis, `tests/` para testes.

## Regras de seguranca

- Nao apague checkpoints, datasets ou artefatos sem pedido explicito.
- Nao carregue checkpoints recebidos de origem nao confiavel.
- Ao usar `torch.load`, prefira `weights_only=True` quando o checkpoint nao exigir objetos Python customizados.
- Nao misture dados de treino, validacao e teste sem registrar a mudanca no manifesto dos splits.

## Comandos

Instalar dependencias:

```powershell
python -m pip install -r requirements.txt
```

Rodar testes:

```powershell
python -m pytest
```

Validar dataset:

```powershell
python -m scripts.check_dataset_quality
```

Treinar tokenizer BPE:

```powershell
python -m scripts.train_bpe_tokenizer
```

Treinar Transformer BPE:

```powershell
python -m scripts.train_transformer_bpe
```

Gerar texto com checkpoint BPE:

```powershell
python -m scripts.generate_transformer_bpe --prompt "A inteligencia artificial "
```

## Preferencias

- Antes de alterar codigo, entenda o fluxo do pipeline e rode testes quando possivel.
- Em revisoes, separe bugs reais de ideias futuras.
- Mantenha alteracoes pequenas e alinhadas ao carater didatico do projeto.
