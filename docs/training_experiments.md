# Experimentos de treinamento

Este documento define uma matriz inicial de experimentos reproduziveis para o
`mini-llm-pt`. O objetivo e comparar configuracoes de treino sem adicionar
runner, schema JSON ou automacao nova.

Antes de rodar os experimentos, valide o dataset e garanta que o tokenizer BPE
existe:

```powershell
python -m scripts.check_dataset_quality
python -m scripts.train_bpe_tokenizer
```

O BPE usa a biblioteca `tokenizers`, entao o treino do tokenizer e bem mais
rapido que a versao didatica em Python puro mesmo com datasets maiores.

Os caminhos sugeridos para metricas ficam em `artifacts/runs/` e os checkpoints
em `checkpoints/`. Esses diretorios ja sao ignorados pelo Git.

## Ordem recomendada

1. Smoke test curto para confirmar que o ambiente, tokenizer, checkpoint e JSONL
   estao funcionando.
2. Baseline BPE com a configuracao atual.
3. Contexto maior com `--block-size 128`.
4. Modelo maior com `--n-embd 256 --n-layer 4`.

Em CPU, o primeiro ganho deve vir de estabilidade e medicao: `grad_clip`,
scheduler, metricas de tokens por segundo e possibilidade de resume. Aumentar
muito `n_embd`, `n_layer` ou `block_size` antes de ter um baseline confiavel
tende a deixar o ciclo de aprendizado lento.

Smoke test sugerido:

```powershell
python -m scripts.train_transformer_bpe --max-iters 2 --eval-interval 1 --eval-num-batches 1 --checkpoint-path checkpoints/smoke.pt --metrics-path artifacts/runs/smoke.jsonl
python -m scripts.train_transformer_bpe --max-iters 3 --eval-interval 1 --eval-num-batches 1 --resume-from checkpoints/smoke.pt --checkpoint-path checkpoints/smoke_resume.pt --metrics-path artifacts/runs/smoke_resume.jsonl
```

Esse smoke usa os splits reais e ainda precisa tokenizar o dataset inteiro. Em
CPU, se a intencao for apenas validar o loop rapidamente, reduza tambem
`--n-embd`, `--n-layer`, `--batch-size` e use arquivos pequenos em
`--train-path` e `--val-path`.

Argumentos uteis para treinos comparaveis:

```txt
--grad-clip 1.0
--warmup-iters 100
--min-learning-rate 1e-4
--disable-lr-schedule
--resume-from checkpoints/algum_checkpoint.pt
--gradient-accumulation-steps 1
```

## Geracao com menos repeticao

Se a geracao entrar em loops como "espaco vetorial espaco vetorial" ou
"complexidade computacional complexidade computacional", use os controles de
decoding antes de concluir que o treino piorou.

Ponto de partida recomendado:

```powershell
python -m scripts.generate_transformer_bpe --checkpoint-path checkpoints/transformer_bpe_block128_dropout02.pt --prompt "A inteligencia artificial e uma area da computacao que " --temperature 0.7 --top-k 20 --top-p 0.9 --repetition-penalty 1.15 --no-repeat-ngram-size 3 --max-new-tokens 80
```

O que cada parametro faz:

- `--top-p 0.9`: usa nucleus sampling, mantendo os tokens mais provaveis ate
  somarem 90% da probabilidade.
- `--repetition-penalty 1.15`: reduz a chance de repetir tokens ja usados.
- `--no-repeat-ngram-size 3`: impede repetir a mesma sequencia de 3 tokens.
- `--temperature 0.7` e `--top-k 20`: aumentam variedade sem liberar o
  vocabulario inteiro.

## Matriz

| experimento | objetivo | checkpoint | metricas | criterio principal |
|---|---|---|---|---|
| `bpe_baseline` | Registrar a configuracao BPE atual como referencia | `checkpoints/transformer_bpe_baseline.pt` | `artifacts/runs/bpe_baseline.jsonl` | Menor `best_val_loss` e geracao qualitativa |
| `bpe_block128` | Medir impacto de contexto maior | `checkpoints/transformer_bpe_block128.pt` | `artifacts/runs/bpe_block128.jsonl` | Comparar `val_loss` com baseline e observar coerencia de texto longo |
| `bpe_larger_model` | Medir impacto de mais capacidade | `checkpoints/transformer_bpe_larger_model.pt` | `artifacts/runs/bpe_larger_model.jsonl` | Melhor `val_loss` sem overfitting evidente |
| `char_vs_bpe` | Comparar caminho por caractere com BPE conservador | `checkpoints/transformer.pt` e `checkpoints/transformer_bpe_char_comparison.pt` | `artifacts/runs/bpe_char_comparison.jsonl` | Comparacao qualitativa; loss nao e diretamente equivalente |

## Experimento 1: baseline BPE atual

Objetivo: criar uma referencia com os defaults atuais de
`train_transformer_bpe.py`.

```powershell
python -m scripts.train_transformer_bpe --device auto --checkpoint-path checkpoints/transformer_bpe_baseline.pt --metrics-path artifacts/runs/bpe_baseline.jsonl
python -m scripts.evaluate_generation_bpe --checkpoint-path checkpoints/transformer_bpe_baseline.pt --output-path artifacts/evaluations/bpe_baseline_generation.txt
```

Criterio de comparacao:

- `best_val_loss` no evento `end` do JSONL;
- estabilidade entre `train_loss` e `val_loss`;
- qualidade das respostas no relatorio qualitativo.

## Experimento 2: contexto maior

Objetivo: testar se `block_size=128` melhora uso de contexto em prompts maiores.

```powershell
python -m scripts.train_transformer_bpe --device auto --block-size 128 --checkpoint-path checkpoints/transformer_bpe_block128.pt --metrics-path artifacts/runs/bpe_block128.jsonl
python -m scripts.evaluate_generation_bpe --checkpoint-path checkpoints/transformer_bpe_block128.pt --output-path artifacts/evaluations/bpe_block128_generation.txt
```

Criterio de comparacao:

- comparar `best_val_loss` contra `bpe_baseline`;
- verificar se o treino ficou muito mais lento;
- comparar geracoes em prompts que dependem de mais contexto.

## Experimento 3: modelo maior

Objetivo: testar se aumentar capacidade melhora validacao sem overfitting claro.

```powershell
python -m scripts.train_transformer_bpe --device auto --n-embd 256 --n-layer 4 --checkpoint-path checkpoints/transformer_bpe_larger_model.pt --metrics-path artifacts/runs/bpe_larger_model.jsonl
python -m scripts.evaluate_generation_bpe --checkpoint-path checkpoints/transformer_bpe_larger_model.pt --output-path artifacts/evaluations/bpe_larger_model_generation.txt
```

Criterio de comparacao:

- `best_val_loss` menor que o baseline;
- diferenca entre `train_loss` e `val_loss` sem aumento persistente de
  overfitting;
- custo de treino aceitavel para a maquina local.

## Experimento 4: comparacao conservadora char vs BPE

Objetivo: comparar o caminho por caractere com um BPE configurado de forma mais
proxima ao Transformer por caractere.

Treino por caractere:

```powershell
python -m scripts.train_transformer
python -m scripts.generate_transformer --checkpoint-path checkpoints/transformer.pt --prompt "A inteligencia artificial" --max-new-tokens 120
```

Treino BPE conservador:

```powershell
python -m scripts.train_transformer_bpe --device auto --block-size 64 --batch-size 16 --max-iters 5000 --eval-interval 500 --n-embd 64 --n-layer 2 --checkpoint-path checkpoints/transformer_bpe_char_comparison.pt --metrics-path artifacts/runs/bpe_char_comparison.jsonl
python -m scripts.generate_transformer_bpe --checkpoint-path checkpoints/transformer_bpe_char_comparison.pt --prompt "A inteligencia artificial " --max-new-tokens 120
```

Criterio de comparacao:

- comparar qualidade de geracao com prompts semelhantes;
- comparar tempo de treino percebido;
- nao tratar `val_loss` como comparacao direta, porque caractere e BPE usam
  unidades de tokenizacao diferentes.

## Observacoes

- Em CPU, `bpe_block128` e `bpe_larger_model` podem ser lentos. Para smoke test,
  reduza temporariamente `--max-iters`, mantendo os nomes de checkpoint e
  metricas do experimento.
- Para comparar resultados, use sempre o mesmo `--seed` quando alterar apenas um
  hiperparametro.
- Nao versionar os checkpoints, metricas JSONL ou relatorios gerados.
