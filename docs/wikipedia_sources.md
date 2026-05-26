# Fontes da Wikipedia para Dataset v0.1/v0.2

Este arquivo reúne links da Wikipedia em português que podem ser usados como fontes candidatas para ampliar o dataset do projeto `mini-llm-pt`.

Mesmo que o modelo seja privado, é importante manter rastreabilidade das fontes usadas.

## Observação de licença

A maior parte dos textos da Wikimedia/Wikipedia é disponibilizada sob licença Creative Commons Attribution-ShareAlike, geralmente CC BY-SA 4.0, com exigência de atribuição e compartilhamento sob licença compatível quando houver redistribuição de derivados.

Para o projeto, registre sempre:

```txt
fonte
url
licença
data de acesso
tipo de limpeza aplicada
```

## Recomendação de uso

Não copie tudo de uma vez. Comece com poucos artigos, limpe bem o texto e observe o impacto em `train loss`, `val loss` e geração.

Fluxo recomendado:

```txt
data/raw/wikipedia/
→ data/processed/wikipedia/
→ data/splits/train.txt
→ data/splits/val.txt
→ data/splits/test.txt
```

## Observação sobre os links

Esta é uma lista candidata. Alguns artigos podem redirecionar para outro título, mudar de nome ou não ter conteúdo suficiente em português. Antes de colocar no dataset, abra o link, confirme o conteúdo e registre a URL final usada.

---

# Inteligência artificial e aprendizado de máquina

- [Inteligência artificial](https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial)
- [Aprendizado de máquina](https://pt.wikipedia.org/wiki/Aprendizado_de_m%C3%A1quina)
- [Aprendizagem profunda](https://pt.wikipedia.org/wiki/Aprendizagem_profunda)
- [Rede neural artificial](https://pt.wikipedia.org/wiki/Rede_neural_artificial)
- [Rede neural](https://pt.wikipedia.org/wiki/Rede_Neural)
- [Mineração de dados](https://pt.wikipedia.org/wiki/Minera%C3%A7%C3%A3o_de_dados)
- [Ciência de dados](https://pt.wikipedia.org/wiki/Ci%C3%AAncia_de_dados)
- [Reconhecimento de padrões](https://pt.wikipedia.org/wiki/Reconhecimento_de_padr%C3%B5es)
- [Visão computacional](https://pt.wikipedia.org/wiki/Vis%C3%A3o_computacional)
- [Sistema especialista](https://pt.wikipedia.org/wiki/Sistema_especialista)
- [Aprendizagem supervisionada](https://pt.wikipedia.org/wiki/Aprendizagem_supervisionada)
- [Aprendizagem não supervisionada](https://pt.wikipedia.org/wiki/Aprendizagem_n%C3%A3o_supervisionada)
- [Aprendizagem por reforço](https://pt.wikipedia.org/wiki/Aprendizagem_por_refor%C3%A7o)
- [Classificação estatística](https://pt.wikipedia.org/wiki/Classifica%C3%A7%C3%A3o_estat%C3%ADstica)
- [Agrupamento de dados](https://pt.wikipedia.org/wiki/Agrupamento_de_dados)
- [Árvore de decisão](https://pt.wikipedia.org/wiki/%C3%81rvore_de_decis%C3%A3o)
- [Floresta aleatória](https://pt.wikipedia.org/wiki/Floresta_aleat%C3%B3ria)
- [Máquina de vetores de suporte](https://pt.wikipedia.org/wiki/M%C3%A1quina_de_vetores_de_suporte)
- [K-means](https://pt.wikipedia.org/wiki/K-means)
- [Perceptron](https://pt.wikipedia.org/wiki/Perceptron)
- [Retropropagação](https://pt.wikipedia.org/wiki/Retropropaga%C3%A7%C3%A3o)
- [Gradiente descendente](https://pt.wikipedia.org/wiki/Gradiente_descendente)
- [Função de ativação](https://pt.wikipedia.org/wiki/Fun%C3%A7%C3%A3o_de_ativa%C3%A7%C3%A3o)
- [Sobreajuste](https://pt.wikipedia.org/wiki/Sobreajuste)
- [Validação cruzada](https://pt.wikipedia.org/wiki/Valida%C3%A7%C3%A3o_cruzada)

# Linguagem natural e texto

- [Processamento de linguagem natural](https://pt.wikipedia.org/wiki/Processamento_de_linguagem_natural)
- [Entendimento de linguagem natural](https://pt.wikipedia.org/wiki/Entendimento_de_linguagem_natural)
- [Linguística computacional](https://pt.wikipedia.org/wiki/Lingu%C3%ADstica_computacional)
- [Recuperação de informação](https://pt.wikipedia.org/wiki/Recupera%C3%A7%C3%A3o_de_informa%C3%A7%C3%A3o)
- [Extração de informação](https://pt.wikipedia.org/wiki/Extra%C3%A7%C3%A3o_de_informa%C3%A7%C3%A3o)
- [Tradução automática](https://pt.wikipedia.org/wiki/Tradu%C3%A7%C3%A3o_autom%C3%A1tica)
- [Reconhecimento de fala](https://pt.wikipedia.org/wiki/Reconhecimento_de_fala)
- [Síntese de fala](https://pt.wikipedia.org/wiki/S%C3%ADntese_de_fala)
- [Corpus linguístico](https://pt.wikipedia.org/wiki/Corpus_lingu%C3%ADstico)
- [Tokenização](https://pt.wikipedia.org/wiki/Tokeniza%C3%A7%C3%A3o)
- [Lematização](https://pt.wikipedia.org/wiki/Lematiza%C3%A7%C3%A3o)
- [Radicalização](https://pt.wikipedia.org/wiki/Radicaliza%C3%A7%C3%A3o)
- [Análise sintática](https://pt.wikipedia.org/wiki/Sintaxe)
- [Análise semântica](https://pt.wikipedia.org/wiki/Sem%C3%A2ntica)
- [Semântica vetorial](https://pt.wikipedia.org/wiki/Sem%C3%A2ntica_vetorial)
- [Word embedding](https://pt.wikipedia.org/wiki/Word_embedding)
- [TF-IDF](https://pt.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Saco de palavras](https://pt.wikipedia.org/wiki/Saco_de_palavras)
- [N-grama](https://pt.wikipedia.org/wiki/N-grama)
- [Modelo oculto de Markov](https://pt.wikipedia.org/wiki/Modelo_oculto_de_Markov)

# Transformers e modelos de linguagem

- [Transformador (arquitetura de aprendizagem profunda)](https://pt.wikipedia.org/wiki/Transformador_%28arquitetura_de_aprendizagem_profunda%29)
- [ChatGPT](https://pt.wikipedia.org/wiki/ChatGPT)
- [Modelo de linguagem](https://pt.wikipedia.org/wiki/Modelo_de_linguagem)
- [Modelo de linguagem grande](https://pt.wikipedia.org/wiki/Modelo_de_linguagem_grande)
- [GPT](https://pt.wikipedia.org/wiki/Generative_Pre-trained_Transformer)
- [BERT](https://pt.wikipedia.org/wiki/BERT_%28modelo_de_linguagem%29)
- [Atenção (aprendizado de máquina)](https://pt.wikipedia.org/wiki/Aten%C3%A7%C3%A3o_%28aprendizado_de_m%C3%A1quina%29)
- [Rede neural recorrente](https://pt.wikipedia.org/wiki/Rede_neural_recorrente)
- [Memória de longo curto-prazo](https://pt.wikipedia.org/wiki/Mem%C3%B3ria_de_longo_curto-prazo)
- [Autoencoder](https://pt.wikipedia.org/wiki/Autoencoder)
- [Codificador-decodificador](https://pt.wikipedia.org/wiki/Codificador-decodificador)
- [Aprendizagem auto-supervisionada](https://pt.wikipedia.org/wiki/Aprendizagem_auto-supervisionada)
- [Aprendizagem por transferência](https://pt.wikipedia.org/wiki/Aprendizagem_por_transfer%C3%AAncia)
- [Modelo generativo](https://pt.wikipedia.org/wiki/Modelo_g%C3%A9nerativo)
- [Inteligência artificial generativa](https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial_generativa)

# Programação e Python

- [Python](https://pt.wikipedia.org/wiki/Python)
- [Linguagem de programação](https://pt.wikipedia.org/wiki/Linguagem_de_programa%C3%A7%C3%A3o)
- [Programação de computadores](https://pt.wikipedia.org/wiki/Programa%C3%A7%C3%A3o_de_computadores)
- [Programação orientada a objetos](https://pt.wikipedia.org/wiki/Programa%C3%A7%C3%A3o_orientada_a_objetos)
- [Programação funcional](https://pt.wikipedia.org/wiki/Programa%C3%A7%C3%A3o_funcional)
- [Interpretador](https://pt.wikipedia.org/wiki/Interpretador)
- [Compilador](https://pt.wikipedia.org/wiki/Compilador)
- [Biblioteca (computação)](https://pt.wikipedia.org/wiki/Biblioteca_%28computa%C3%A7%C3%A3o%29)
- [API](https://pt.wikipedia.org/wiki/API)
- [Código-fonte](https://pt.wikipedia.org/wiki/C%C3%B3digo-fonte)
- [Software livre](https://pt.wikipedia.org/wiki/Software_livre)
- [Código aberto](https://pt.wikipedia.org/wiki/C%C3%B3digo_aberto)
- [Git](https://pt.wikipedia.org/wiki/Git)
- [GitHub](https://pt.wikipedia.org/wiki/GitHub)
- [Ambiente de desenvolvimento integrado](https://pt.wikipedia.org/wiki/Ambiente_de_desenvolvimento_integrado)
- [Depuração](https://pt.wikipedia.org/wiki/Depura%C3%A7%C3%A3o)
- [Teste de software](https://pt.wikipedia.org/wiki/Teste_de_software)
- [Refatoração](https://pt.wikipedia.org/wiki/Refatora%C3%A7%C3%A3o)

# Algoritmos e estruturas de dados

- [Algoritmo](https://pt.wikipedia.org/wiki/Algoritmo)
- [Estrutura de dados](https://pt.wikipedia.org/wiki/Estrutura_de_dados)
- [Lista ligada](https://pt.wikipedia.org/wiki/Lista_ligada)
- [Pilha (informática)](https://pt.wikipedia.org/wiki/Pilha_%28inform%C3%A1tica%29)
- [Fila (informática)](https://pt.wikipedia.org/wiki/Fila_%28inform%C3%A1tica%29)
- [Árvore (estrutura de dados)](https://pt.wikipedia.org/wiki/%C3%81rvore_%28estrutura_de_dados%29)
- [Tabela de dispersão](https://pt.wikipedia.org/wiki/Tabela_de_dispers%C3%A3o)
- [Busca binária](https://pt.wikipedia.org/wiki/Pesquisa_bin%C3%A1ria)
- [Ordenação](https://pt.wikipedia.org/wiki/Algoritmo_de_ordena%C3%A7%C3%A3o)
- [Complexidade computacional](https://pt.wikipedia.org/wiki/Complexidade_computacional)
- [Notação Big O](https://pt.wikipedia.org/wiki/Nota%C3%A7%C3%A3o_Big_O)
- [Recursão](https://pt.wikipedia.org/wiki/Recurs%C3%A3o_%28ci%C3%AAncia_da_computa%C3%A7%C3%A3o%29)
- [Programação dinâmica](https://pt.wikipedia.org/wiki/Programa%C3%A7%C3%A3o_din%C3%A2mica)
- [Algoritmo guloso](https://pt.wikipedia.org/wiki/Algoritmo_guloso)
- [Busca em largura](https://pt.wikipedia.org/wiki/Busca_em_largura)
- [Busca em profundidade](https://pt.wikipedia.org/wiki/Busca_em_profundidade)
- [Grafo](https://pt.wikipedia.org/wiki/Grafo)
- [Teoria dos grafos](https://pt.wikipedia.org/wiki/Teoria_dos_grafos)
- [Dijkstra](https://pt.wikipedia.org/wiki/Algoritmo_de_Dijkstra)

# Matemática para IA

- [Álgebra linear](https://pt.wikipedia.org/wiki/%C3%81lgebra_linear)
- [Matriz](https://pt.wikipedia.org/wiki/Matriz_%28matem%C3%A1tica%29)
- [Vetor](https://pt.wikipedia.org/wiki/Vetor_%28matem%C3%A1tica%29)
- [Produto escalar](https://pt.wikipedia.org/wiki/Produto_escalar)
- [Produto vetorial](https://pt.wikipedia.org/wiki/Produto_vetorial)
- [Espaço vetorial](https://pt.wikipedia.org/wiki/Espa%C3%A7o_vetorial)
- [Autovalores e autovetores](https://pt.wikipedia.org/wiki/Autovalores_e_autovetores)
- [Decomposição em valores singulares](https://pt.wikipedia.org/wiki/Decomposi%C3%A7%C3%A3o_em_valores_singulares)
- [Cálculo](https://pt.wikipedia.org/wiki/C%C3%A1lculo)
- [Derivada](https://pt.wikipedia.org/wiki/Derivada)
- [Derivada parcial](https://pt.wikipedia.org/wiki/Derivada_parcial)
- [Gradiente](https://pt.wikipedia.org/wiki/Gradiente)
- [Otimização matemática](https://pt.wikipedia.org/wiki/Otimiza%C3%A7%C3%A3o_matem%C3%A1tica)
- [Probabilidade](https://pt.wikipedia.org/wiki/Probabilidade)
- [Distribuição de probabilidade](https://pt.wikipedia.org/wiki/Distribui%C3%A7%C3%A3o_de_probabilidade)
- [Distribuição normal](https://pt.wikipedia.org/wiki/Distribui%C3%A7%C3%A3o_normal)
- [Estatística](https://pt.wikipedia.org/wiki/Estat%C3%ADstica)
- [Inferência estatística](https://pt.wikipedia.org/wiki/Infer%C3%AAncia_estat%C3%ADstica)
- [Entropia da informação](https://pt.wikipedia.org/wiki/Entropia_da_informa%C3%A7%C3%A3o)
- [Teoria da informação](https://pt.wikipedia.org/wiki/Teoria_da_informa%C3%A7%C3%A3o)

# Ciência da computação

- [Ciência da computação](https://pt.wikipedia.org/wiki/Ci%C3%AAncia_da_computa%C3%A7%C3%A3o)
- [Teoria da computação](https://pt.wikipedia.org/wiki/Teoria_da_computa%C3%A7%C3%A3o)
- [Máquina de Turing](https://pt.wikipedia.org/wiki/M%C3%A1quina_de_Turing)
- [Computabilidade](https://pt.wikipedia.org/wiki/Computabilidade)
- [Lógica em ciência da computação](https://pt.wikipedia.org/wiki/L%C3%B3gica_em_ci%C3%AAncia_da_computa%C3%A7%C3%A3o)
- [Sistema operacional](https://pt.wikipedia.org/wiki/Sistema_operacional)
- [Banco de dados](https://pt.wikipedia.org/wiki/Banco_de_dados)
- [Redes de computadores](https://pt.wikipedia.org/wiki/Rede_de_computadores)
- [Engenharia de software](https://pt.wikipedia.org/wiki/Engenharia_de_software)
- [Arquitetura de computadores](https://pt.wikipedia.org/wiki/Arquitetura_de_computadores)
- [Computação paralela](https://pt.wikipedia.org/wiki/Computa%C3%A7%C3%A3o_paralela)
- [Computação distribuída](https://pt.wikipedia.org/wiki/Computa%C3%A7%C3%A3o_distribu%C3%ADda)
- [Computação em nuvem](https://pt.wikipedia.org/wiki/Computa%C3%A7%C3%A3o_em_nuvem)
- [Segurança da informação](https://pt.wikipedia.org/wiki/Seguran%C3%A7a_da_informa%C3%A7%C3%A3o)
- [Criptografia](https://pt.wikipedia.org/wiki/Criptografia)
- [Inteligência computacional](https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_computacional)

# Dados, informação e estatística aplicada

- [Dado](https://pt.wikipedia.org/wiki/Dado)
- [Informação](https://pt.wikipedia.org/wiki/Informa%C3%A7%C3%A3o)
- [Base de dados](https://pt.wikipedia.org/wiki/Base_de_dados)
- [Amostragem](https://pt.wikipedia.org/wiki/Amostragem)
- [Regressão linear](https://pt.wikipedia.org/wiki/Regress%C3%A3o_linear)
- [Regressão logística](https://pt.wikipedia.org/wiki/Regress%C3%A3o_log%C3%ADstica)
- [Classificação estatística](https://pt.wikipedia.org/wiki/Classifica%C3%A7%C3%A3o_estat%C3%ADstica)
- [Análise de dados](https://pt.wikipedia.org/wiki/An%C3%A1lise_de_dados)
- [Visualização de dados](https://pt.wikipedia.org/wiki/Visualiza%C3%A7%C3%A3o_de_dados)
- [Big data](https://pt.wikipedia.org/wiki/Big_data)
- [Armazém de dados](https://pt.wikipedia.org/wiki/Armaz%C3%A9m_de_dados)
- [ETL](https://pt.wikipedia.org/wiki/Extract,_transform,_load)
- [Metadados](https://pt.wikipedia.org/wiki/Metadados)
- [Qualidade de dados](https://pt.wikipedia.org/wiki/Qualidade_de_dados)

# Hardware, GPU e computação para ML

- [Unidade de processamento gráfico](https://pt.wikipedia.org/wiki/Unidade_de_processamento_gr%C3%A1fico)
- [CUDA](https://pt.wikipedia.org/wiki/CUDA)
- [Processador](https://pt.wikipedia.org/wiki/Unidade_central_de_processamento)
- [Memória de acesso aleatório](https://pt.wikipedia.org/wiki/Mem%C3%B3ria_de_acesso_aleat%C3%B3rio)
- [Memória de computador](https://pt.wikipedia.org/wiki/Mem%C3%B3ria_de_computador)
- [Supercomputador](https://pt.wikipedia.org/wiki/Supercomputador)
- [Cluster de computadores](https://pt.wikipedia.org/wiki/Cluster_de_computadores)
- [Computação de alto desempenho](https://pt.wikipedia.org/wiki/Computa%C3%A7%C3%A3o_de_alto_desempenho)

# Ética, sociedade e impacto da IA

- [Ética na inteligência artificial](https://pt.wikipedia.org/wiki/%C3%89tica_na_intelig%C3%AAncia_artificial)
- [Viés algorítmico](https://pt.wikipedia.org/wiki/Vi%C3%A9s_algor%C3%ADtmico)
- [Privacidade](https://pt.wikipedia.org/wiki/Privacidade)
- [Proteção de dados pessoais](https://pt.wikipedia.org/wiki/Prote%C3%A7%C3%A3o_de_dados_pessoais)
- [Lei Geral de Proteção de Dados Pessoais](https://pt.wikipedia.org/wiki/Lei_Geral_de_Prote%C3%A7%C3%A3o_de_Dados_Pessoais)
- [Direitos autorais](https://pt.wikipedia.org/wiki/Direito_autoral)
- [Licença Creative Commons](https://pt.wikipedia.org/wiki/Licen%C3%A7as_Creative_Commons)
- [Desinformação](https://pt.wikipedia.org/wiki/Desinforma%C3%A7%C3%A3o)
- [Automação](https://pt.wikipedia.org/wiki/Automa%C3%A7%C3%A3o)

# História e fundamentos da computação e IA

- [História da inteligência artificial](https://pt.wikipedia.org/wiki/Hist%C3%B3ria_da_intelig%C3%AAncia_artificial)
- [Alan Turing](https://pt.wikipedia.org/wiki/Alan_Turing)
- [Teste de Turing](https://pt.wikipedia.org/wiki/Teste_de_Turing)
- [John McCarthy](https://pt.wikipedia.org/wiki/John_McCarthy)
- [Marvin Minsky](https://pt.wikipedia.org/wiki/Marvin_Minsky)
- [Geoffrey Hinton](https://pt.wikipedia.org/wiki/Geoffrey_Hinton)
- [Yann LeCun](https://pt.wikipedia.org/wiki/Yann_LeCun)
- [Yoshua Bengio](https://pt.wikipedia.org/wiki/Yoshua_Bengio)
- [Donald Knuth](https://pt.wikipedia.org/wiki/Donald_Knuth)
- [Ada Lovelace](https://pt.wikipedia.org/wiki/Ada_Lovelace)

# Educação técnica e conhecimento geral útil

- [Método científico](https://pt.wikipedia.org/wiki/M%C3%A9todo_cient%C3%ADfico)
- [Conhecimento](https://pt.wikipedia.org/wiki/Conhecimento)
- [Aprendizagem](https://pt.wikipedia.org/wiki/Aprendizagem)
- [Educação](https://pt.wikipedia.org/wiki/Educa%C3%A7%C3%A3o)
- [Raciocínio lógico](https://pt.wikipedia.org/wiki/Racioc%C3%ADnio_l%C3%B3gico)
- [Lógica](https://pt.wikipedia.org/wiki/L%C3%B3gica)
- [Inferência](https://pt.wikipedia.org/wiki/Infer%C3%AAncia)
- [Problema](https://pt.wikipedia.org/wiki/Problema)
- [Solução de problemas](https://pt.wikipedia.org/wiki/Solu%C3%A7%C3%A3o_de_problemas)

---

# Sugestão para Dataset v0.1

Comece com 10 a 15 artigos técnicos, não com todos os links.

Prioridade inicial:

```txt
1. Inteligência artificial
2. Aprendizado de máquina
3. Aprendizagem profunda
4. Rede neural artificial
5. Processamento de linguagem natural
6. Transformador
7. Modelo de linguagem grande
8. Python
9. Algoritmo
10. Álgebra linear
11. Gradiente descendente
12. Ciência de dados
13. Corpus linguístico
14. Word embedding
15. Teoria da informação
```

Depois de processar esses textos, treine novamente e compare:

```txt
train loss
val loss
qualidade da geração
quantidade de <unk>
tamanho do vocabulário
número total de caracteres/tokens
```

# Sugestão para Dataset v0.2

Depois que o Dataset v0.1 estiver funcionando, expanda para 40 a 80 artigos.

Priorize:

```txt
IA e aprendizado de máquina
NLP e linguagem
programação
algoritmos
matemática para IA
ética e impacto da IA
```

Evite misturar muitos temas distantes no começo. O objetivo é fazer o modelo aprender um domínio técnico coerente em português.
