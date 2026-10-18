# Benchmarks e escolhas

## De onde veio o MCQ

Montei a base na aplicação separada **ACartomanteAnnotation**, com Flask e SQLite: dez perguntas para cada uma de dez obras, totalizando 100. Cada anotação guarda pergunta, resposta e evidência em texto, para remapear o trecho quando o chunking muda.

O conjunto cobre A carteira, A Cartomante, A igreja do Diabo, Dom Casmurro, Jogo do Bicho, Memórias Póstumas de Brás Cubas, Missa do Galo, O Alienista, O imortal e Quincas Borba. O MCQ tem cinco alternativas, com uma correta e distratores errados ou ambíguos.

Os CSVs em `AED/` registram a [anotação manual](../AED/benchmark_questions_manual.csv), a [reescrita](../AED/benchmark_questions_reescrito.csv), as [alternativas](../AED/benchmark_questions_mcq.csv) e o [export final](../AED/benchmark_questions_final.csv), consumido por `benchmark.prepare`. Mantêm o mesmo `query_id` e status de rascunho; não estou presumindo validação externa de uma base artesanal.

A divisão é determinística por obra e dificuldade: **60 DEV e 40 holdout**, seed `20260930`. DEV serve para escolher; holdout, para avaliar depois de congelar a configuração. Retrieval mede a recuperação da evidência; MCQ mede a escolha do gerador com aquele contexto. Acertar a letra não resolve as duas coisas sozinho.

## O que ficou dos testes locais

O retrieval selecionou dez de 30 configurações. Cruzei essas dez com quatro geradores locais: Llama 3, Qwen 3.5 4B, Ministral 3B e Mistral 7B. Foram 2.400 respostas DEV e 1.600 holdout, repetindo as mesmas perguntas; não são milhares de perguntas independentes.

Ficou selecionado **fixed/256, overlap 32, Colibri e Dense + CrossEncoder**. Com Qwen 3.5 4B, o MCQ marcou **44/60 em DEV (73,33%) e 25/40 no holdout (62,50%)**.

Nesta ablação, cada par agrega os mesmos quatro geradores: 240 respostas DEV e 160 holdout.

| Troca, mantendo o restante do par | DEV: antes → depois | Holdout: antes → depois |
| --- | ---: | ---: |
| Fixed → híbrido; 256, Colibri, Dense+CE | 67,50% → 63,33% | 58,75% → 57,50% |
| Colibri → Qwen3 Embedding 0.6B; fixed/256, Dense+CE | 67,50% → 60,83% | 58,75% → 57,50% |
| Dense → Dense+CE; híbrido/1024, Colibri | 58,75% → 57,08% | 50,63% → 50,63% |

Depois de todo o trabalho com capítulos, o híbrido perdeu aquele par para o fixo. Os números tiveram a delicadeza de não respeitar meu esforço. Isso orientou a escolha, sem provar superioridade geral de um componente.

No A/B DEV, acrescentar enrichment mudou a média de **60,04% para 61,67%**. O ganho de 1,625 ponto percentual teve intervalo de 95% entre −1,083 e +4,75 pontos, por bootstrap de `query_id` com 2.000 reamostragens, seed 42. O efeito variou por modelo e o intervalo inclui zero.

Mantive os resumos como apoio limitado. O A/B acrescentava contexto; o chat divide o orçamento entre texto e resumos. A AWS usa Qwen3 32B, guard e histórico. **Os 62,50% locais não medem a acurácia do chat publicado.**

## Executar o MCQ na AWS

Com a [instalação AWS](REPRODUCAO_AWS.md) e uma release publicada, abra **Actions → AWS MCQ benchmark → Run workflow**, na `main`:

| Campo | Escolha |
| --- | --- |
| `operation` | `run` para avaliar; `report` para refazer relatórios sem inferência |
| `model` | `qwen3-32b` ou `llama4-scout` |
| `split` | `dev`; holdout só depois de congelar as escolhas |
| `limit_queries` | `2` para conferir o caminho; `0` para todo o split |
| `run_id` | Vazio cria uma run; em `report`, informe o ID salvo |

Qwen usa Mantle; Scout, Converse. Ambos dependem do acesso e das quotas da conta. O workflow usa OIDC com a role ETL, a recuperação atual e o enrichment publicado. Não reconstrói índices nem executa o sweep de retrieval.

O nome identifica modelo, commit e split. Os artefatos guardam SHA completo e configuração. Para comparar, preserve release, perguntas e protocolo. Para retomar pendências, use **Re-run jobs** na execução original, preservando também modelo, SHA, run, split e limite.

Baixe **Artifacts → benchmark-…**, extraia e abra `index.html`: o painel reúne acurácia, latência p95, tokens e custo por run. `runs/` contém páginas individuais; `report.html`, a execução atual. No S3 privado: painel em `mcq/current/index.html` e resultados/checkpoints em `mcq/current/runs/<run_id>/`. O custo é estimado, não a fatura.

## Conferir os experimentos originais

- [Seleção das dez configurações](../reports/retrieval_retrieval_dev_20261001T042919Z_final10.json).
- [Retrieval DEV/holdout](../reports/retrieval_retrieval_dev_20261001T042919Z__retrieval_holdout_20261002T144006Z.html).
- [MCQ baseline DEV/holdout](../reports/mcq_mcq_dev_20261001T212056Z__mcq_holdout_20261002T145157Z.html).
- [A/B do enrichment](../reports/mcq_enrichment_mcq_enriched_dev_v1.html).

Os notebooks [ExploreCorpus](../AED/ExploreCorpus.ipynb), [extração de capítulos](../AED/ExploreEstrategiaExtracaoCapitulos.ipynb) e [classificação](../AED/ClassificacaoObras.ipynb) mostram a investigação. Os runners históricos ficam em `benchmark/prepare.py`, `run_retrieval.py` e `run_mcq.py`; repetir exatamente os números exige também os snapshots e digests registrados, não apenas executar o código atual.
