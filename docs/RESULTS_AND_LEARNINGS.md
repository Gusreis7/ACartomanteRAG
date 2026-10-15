# Resultados, decisões e o que aprendi no caminho

Quando comecei a investigar os capítulos das obras, minha hipótese era aproveitar as divisões semânticas que já existiam nos textos para fazer um RAG melhor. Parecia um caminho bastante razoável: se a obra já está organizada, por que ignorar essa estrutura?

Ao longo do projeto, porém, precisei separar o que fazia sentido como hipótese do que os experimentos realmente sustentavam. Este documento registra essa passagem: como comparei as alternativas, o que entrou na configuração atual e quais conclusões ainda não consigo tirar dos resultados.

Os notebooks guardam a investigação que levou a essas perguntas: [exploração do corpus](../AED/ExploreCorpus.ipynb), [extração de capítulos](../AED/ExploreEstrategiaExtracaoCapitulos.ipynb) e [classificação das obras](../AED/ClassificacaoObras.ipynb). A [arquitetura](RUNTIME_ARCHITECTURE.md) descreve como as escolhas foram incorporadas à aplicação.

## Primeiro, criar uma forma de comparar

Eu não tinha um conjunto pronto de perguntas e evidências anotadas para este acervo. A base de avaliação foi construída e revisada na aplicação separada ACartomanteAnnotation, com Flask e SQLite, partindo de 10 obras com 10 perguntas por obra. As evidências foram mantidas como texto, para permitir seu remapeamento quando eu mudasse a divisão em chunks.

Separei 60 questões em DEV e 40 em holdout. Usei DEV para comparar e selecionar alternativas; o holdout ficou como um recorte separado para observar o comportamento fora da seleção. Esse cuidado não transforma a amostra em uma avaliação abrangente da literatura machadiana, mas dá um critério melhor do que escolher pela impressão de algumas respostas no chat.

Também separei duas avaliações. No retrieval, me interessa saber se os trechos relevantes chegam à recuperação e em que posição aparecem. Na geração, avalio se o modelo consegue usar o contexto para responder às questões de múltipla escolha. Uma recuperação boa pode ser mal aproveitada pelo modelo; uma resposta correta isolada também não prova que o processo de recuperação foi bom.

Para reduzir a matriz que seguiria à geração, selecionei 10 configurações entre 30 elegíveis pelo Balanced Rank Score. O critério combina percentis de ranking de quatro famílias, com peso igual: hits, cobertura da evidência, ranking e acerto da obra. Isso evita decidir olhando apenas uma métrica favorável a uma configuração. As definições e a seleção estão no [registro das dez configurações](../reports/retrieval_retrieval_dev_20261001T042919Z_final10.json).

Na etapa MCQ, essas 10 configurações foram cruzadas com 4 modelos. Portanto, 2.400 gerações em DEV continuam sendo avaliações sobre 60 perguntas; no holdout, 1.600 gerações correspondem a 40 perguntas. Repetir uma pergunta com outros modelos e parâmetros não aumenta a quantidade de perguntas independentes da base.

## O que entrou no runtime e o que os números representam

O [relatório MCQ de DEV e holdout](../reports/mcq_mcq_dev_20261001T212056Z__mcq_holdout_20261002T145157Z.html) apresenta estes recortes da matriz baseline:

| Recorte | Acurácia | O que está sendo agregado |
|---|---:|---|
| Matriz completa em DEV. | 60,04% | 10 configurações × 4 modelos × 60 perguntas: 2.400 gerações. |
| Matriz completa no holdout. | 54,19% | 10 configurações × 4 modelos × 40 perguntas: 1.600 gerações. |
| Fixed/256 + Colibri + Dense+CE em DEV. | 67,50% | Os 4 modelos com esse retrieval: 240 gerações. |
| Mesmo retrieval no holdout. | 58,75% | Os 4 modelos com esse retrieval: 160 gerações. |
| Qwen 3.5 4B com esse retrieval em DEV. | 73,33% | 44 acertos em 60 perguntas, no baseline. |
| Qwen 3.5 4B com esse retrieval no holdout. | 62,50% | 25 acertos em 40 perguntas, no baseline. |

O registro de promoção da configuração Qwen + fixed/256 + Colibri + Dense+CE apontava **73,3% nas 60 questões DEV**. Na revisão, conferi esse resultado nos registros individuais dos sub-relatórios DEV e holdout, que estão embutidos no HTML consolidado. Filtrando `model=qwen3.5:4b` e `config_id=fixed_c256_o32__dense_cross_encoder_colibri`, encontrei os 44 acertos em DEV e os 25 no holdout.

Essa foi a maior acurácia entre as 40 combinações de modelo e retrieval avaliadas naquele DEV. É esse recorte que sustenta a seleção, sem transformar a escolha em uma afirmação de superioridade geral.

Consigo, portanto, avaliar essa combinação de maneira mais específica: ela caiu de 73,33% em DEV para 62,50% no holdout, uma diferença de 10,83 pontos percentuais entre esses recortes. Os 67,50% são outra agregação, a média dos quatro modelos com aquele retrieval em DEV; os 54,19% representam a matriz inteira no holdout. Misturar essas unidades de análise mudaria a história que os números contam.

Há ainda uma diferença entre esse benchmark e o chat. A avaliação MCQ baseline usou temperatura 0, saída de até 256 tokens, `num_ctx=8192`, contexto de até 18.000 caracteres e uma geração por vez. O runtime atual usa outros limites e inclui enrichment. Assim, os 62,50% são um resultado holdout verificado da combinação no benchmark baseline, não uma medição da aplicação conversacional atual. Os manifests e resultados dos runs permanecem no RustFS como referência da execução.

A configuração que ficou selecionada no projeto é esta:

| Componente | Escolha atual |
|---|---|
| Silver | `fixed`, tokenizer RoBERTaCrawlPT, 256 tokens e overlap de 32. |
| Embedding textual | `tardellirs/colibri-embed-ptbr`, 768 dimensões, distância cosine e inferência float32. |
| Retrieval | Dense com 25 candidatos, seguido de `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, retornando até 10. |
| Índice selecionado | `idx_colibri-embed-ptbr_2b279b323307`. |
| Geração | Ollama com `qwen3.5:4b`, reasoning desativado e temperatura 0,1. |
| Enrichment | Ativo como apoio, top 3, orçamento limitado e `fail_open=true`. |

Vejam que o chunking fixo permaneceu na aplicação mesmo depois de todo o trabalho com capítulos. Isso não elimina o valor da investigação: ela revelou problemas da fonte, ampliou a classificação e deixou a estratégia híbrida disponível. O que não faria sentido seria impor o híbrido somente porque investi tempo nele. Para a escolha do produto, segui a combinação selecionada nos experimentos, mantendo explícito qual avaliação sustenta essa decisão.

Dense + CrossEncoder e Colibri também entram como uma combinação escolhida neste corpus e nesses testes. O reranking acrescenta trabalho em CPU e latência; limitar a entrada a 25 candidatos controla parte desse custo. Não extraí daí uma regra de que esse embedder ou essa estratégia sempre será superior em outro acervo.

## A aposta nos resumos auxiliares

Uma hipótese seguinte foi usar resumos para ajudar o modelo a reconhecer eventos e relações narrativas que podem ficar dispersos nos trechos recuperados. Essa foi a motivação do enrichment. O cuidado necessário era que o material adicional continuava sendo um derivado produzido por modelo, sujeito às suas próprias falhas.

Nos ajustes do projeto, contratos de saída mais interpretativos aumentaram as dificuldades e o custo do structured output. Mantive então um esquema menor: `resumo + eventos` por unidade e `resumo_geral + eventos_principais` na consolidação da obra. A intenção foi pedir uma tarefa mais delimitada, que eu conseguisse validar e operar com os modelos locais, antes de acrescentar novas camadas interpretativas.

Depois, comparei baseline e enrichment nas mesmas 2.400 combinações de DEV. No [relatório A/B](../reports/mcq_enrichment_mcq_enriched_dev_v1.html), o resultado agregado foi:

| Medida | Resultado |
|---|---:|
| Acurácia baseline → enriquecido. | 60,04% → 61,67%. |
| Diferença. | +1,625 ponto percentual. |
| Intervalo de confiança de 95% do delta. | −1,083 a +4,75 pontos percentuais. |
| Execuções que melhoraram, pioraram e ficaram iguais. | 240, 201 e 1.959, respectivamente. |

O bootstrap foi feito por `query_id`, com 2.000 amostras e seed 42, preservando a pergunta como unidade da reamostragem. O intervalo inclui zero. Portanto, o aumento da média é um sinal modesto neste experimento; ele não comprova um ganho geral do enrichment.

A média também esconde comportamentos diferentes. Llama 3 teve delta de +3,33 pontos percentuais; Qwen 3.5, +3,83; Ministral, +0,83; e Mistral, −1,50. Houve tanto respostas beneficiadas quanto respostas prejudicadas pelo material extra. Mais contexto não equivale automaticamente a contexto mais útil para todos os modelos.

Esse apoio também ocupa espaço: o relatório registra uma média de aproximadamente 5.518 para 5.948 tokens de entrada, cerca de 429 a mais. É um custo observado nessa matriz de testes, não uma previsão de consumo para qualquer conversa na aplicação.

Outro detalhe importa na comparação com o produto atual: o A/B registra o acréscimo dos resumos sem remover o contexto textual baseline. No runtime, estabeleci um orçamento conjunto de 20.000 caracteres, em que o enrichment pode usar no máximo 6.000 caracteres e 25% do total. Com esses parâmetros, o teto efetivo é 5.000. Assim, o experimento A/B ajuda a discutir a hipótese, mas não valida sozinho a configuração final com esse orçamento.

Decidi manter o enrichment como apoio limitado e com `fail_open=true`. Se ele estiver indisponível, o caminho textual pode continuar. Essa é uma decisão operacional e experimental que assumi; não é uma consequência obrigatória de o ganho médio ter sido positivo, nem uma afirmação de que ele melhora toda resposta.

## Algumas decisões que passaram a fazer mais sentido

**Preservar os artefatos e promover a escolha explicitamente.** Como eu estava comparando tamanhos de chunk, tokenizers, embeddings e estratégias, sobrescrever tudo a cada teste dificultaria saber o que gerou um resultado. As camadas versionadas, os manifests e as releases ajudam a reconstruir essa relação. O custo é mais armazenamento e cuidado operacional, mas alterar um YAML sem saber qual índice está carregado seria uma economia ruim para um projeto de experimentação.

**Tratar o índice e o modelo de consulta como um conjunto.** Um novo nome de modelo no arquivo de configuração não converte os vetores de uma collection antiga. A seleção do índice precisa apontar para um artefato compatível, com o corpus e a revisão correspondentes. Colibri está no runtime porque acompanha a escolha atual; trocar o embedder exige o índice adequado e uma nova avaliação.

**Priorizar estabilidade numérica onde houve problema de dtype.** Um dos ajustes registrados foi manter float32 diante de um erro observado com o embedding Qwen. Isso consome mais memória que bfloat16. Também preciso distinguir a precisão usada na inferência dos vetores de saída, que o backend entrega em float32: mudar um desses parâmetros não significa, por si só, que toda collection precise ser reconstruída.

**Deixar o leitor inspecionar o original.** Os trechos `[E]` são as evidências citáveis; os resumos `[R]` ficam no apoio interno. Preferi orientar a resposta a selecionar referências úteis, em vez de enumerar todos os chunks recuperados. Ainda assim, uma citação exibida não prova que a resposta esteja corretamente sustentada por ela. Essa fidelidade precisa ser avaliada, especialmente quando uma persona recebe uma orientação interpretativa.

**Manter a operação proporcional ao experimento.** O monólito modular e um worker simplificam o ambiente local e evitam multiplicar os modelos em memória e os limites de geração. Os jobs e benchmarks reutilizam os módulos sem passar pela API, com configurações próprias por execução. Isso permite experimentar sem misturar a configuração do teste com a do atendimento web. Persistir conversas no Redis ajuda na continuidade, mas não remove os custos de memória, CPU e geração quando eu pensar em escalar.

## Até onde considero que os resultados me levaram

No primeiro notebook, usei acessos às obras como uma aproximação para fazer perguntas sobre popularidade e qualidade e depois discuti os limites daquela métrica. Aqui vale o mesmo cuidado: acertar questões de múltipla escolha é um indicador útil para comparar este conjunto de experimentos, mas não resume a qualidade de uma conversa literária, a fidelidade das citações ou a capacidade de compreender Machado em geral.

Tanto a matriz agregada quanto a combinação Qwen + fixed/256 + Colibri + Dense+CE ficaram abaixo do DEV no holdout. Isso reforça o cuidado com os resultados da seleção, sem permitir atribuir a diferença inteira a uma causa específica. Na combinação escolhida, o resultado holdout continua baseado em apenas 40 perguntas. Da mesma forma, o A/B trouxe uma hipótese de ganho pequeno e variável; ele não fecha a avaliação do produto enriquecido.

Saio desta etapa com uma configuração selecionada, uma forma de rastrear os artefatos e critérios mais claros para investigar uma resposta ruim: olhar a recuperação, o contexto e a geração separadamente. Também ficou um aprendizado que atravessou os notebooks e chegou ao chat: uma estrutura mais elaborada precisa justificar o custo que acrescenta, e os achados podem contrariar a primeira ideia que parecia tão boa.

Este relato consolida os notebooks, os relatórios disponíveis e a configuração presente nos fontes. A revisão da documentação não executou novos benchmarks, chamadas a LLMs nem verificou os volumes da instalação. Os números acima pertencem aos experimentos identificados nos links; não são uma medição nova do runtime atual.
