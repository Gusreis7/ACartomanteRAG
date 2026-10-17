# MCQ da configuração atual

O workflow **AWS MCQ benchmark** avalia a configuração de `config/config.aws.yaml`
e sua base `config.yaml`: modelo e parâmetros de geração, retrieval textual,
enrichment e a release de dados publicada. Hoje isso corresponde ao Qwen3 32B
via Bedrock Mantle, Colibri e dense + cross-encoder. A inferência MCQ usa o
schema Pydantic existente, com os mesmos retries do provider.

O benchmark anterior dependia da seleção de dez configurações de retrieval e
de modelos Ollama locais. Esta execução avalia uma única configuração, sem
exigir essa etapa. As perguntas ainda consultam os índices existentes para
montar o contexto. Não há criação de embeddings do corpus ou de enrichment.

## Primeira execução

1. Aplique o patch e versione as mudanças no Gitea. Aguarde o espelhamento
   da `main` e o sucesso de **CI and AWS deploy**. O Terraform adiciona a
   política `github_mcq` à role de dados existente. Se o apply não tiver
   executado, rode **AWS infrastructure → apply** antes do benchmark.
2. Abra **GitHub → Actions → AWS MCQ benchmark → Run workflow**, na `main`.
3. Use `split=dev`, `limit_queries=2` e `run_id` vazio para a primeira checagem.
4. Para avaliar o DEV inteiro, faça outra execução com `limit_queries=0` e
   `run_id` vazio. Use `holdout` em uma execução separada, após congelar as escolhas.

| Campo | Uso |
| --- | --- |
| `split` | `dev` ou `holdout`; nunca mistura os dois. |
| `limit_queries` | `0` usa todo o split; outro número limita as primeiras questões. |
| `run_id` | Vazio gera `mcq-<split>-<ID GitHub>`. Repetir retoma erros e pendências. |

Também é possível disparar pelo notebook:

```bash
gh workflow run benchmark-mcq.yml --repo Gusreis7/ACartomanteRAG --ref main \
  -f split=dev -f limit_queries=0
```

## Resultados e retomada

O resumo do GitHub mostra o caminho privado no bucket de benchmarks:
`mcq/current/runs/<run_id>/`. A pasta contém configuração sem credenciais,
split determinístico, manifest, respostas/contextos por questão, CSV, Parquet,
resumo e `report.html`. O relatório registra acertos, alternativas ambíguas,
erros, latência, tokens e custo estimado com os preços da configuração.
Para abrir o HTML no notebook, baixe o endereço exibido no resumo com
`aws s3 cp <endereço-S3> ./mcq-report.html --profile terraform`.

Para retomar, mantenha `run_id`, SHA, split e limite. Use **Re-run jobs** na
execução original para preservar o SHA. Respostas válidas são reutilizadas,
inclusive alternativas erradas: a retomada tenta somente erros de execução
ou parsing e questões ainda não processadas. A release imutável original é
preservada mesmo que outra release seja promovida depois. Divergências são
recusadas antes de substituir respostas. Um run incompleto encerra com falha,
mantendo os checkpoints e o relatório parcial para investigação.
Erros fatais de cota/permissão ou três questões com erro consecutivo encerram
o processamento, preservando as demais como pendentes.

## Execução e limites

O runner GitHub calcula somente embeddings de consultas e reranking em CPU;
S3 e Qdrant fornecem os artefatos, e o Bedrock faz a geração. Não são criadas
instâncias ou jobs Glue. Há consumo de Actions, inferência, armazenamento e
requisições S3/Qdrant. O custo informado cobre os tokens retornados pelo provider,
incluindo retries e retomadas registrados; uma interrupção antes do checkpoint
pode deixar uma chamada sem contabilização local.

O MCQ usa o prompt de alternativas, sem histórico ou personas, e uma questão
por vez. A latência inclui preparação do contexto e geração estruturada; não
representa latência de usuários simultâneos na API. Os parâmetros da geração
não são substituídos pelos antigos limites do benchmark Ollama.

O acesso usa `AWS_ETL_ROLE_ARN`, por OIDC na `main`, e o parâmetro SSM da aplicação
para obter a chave Qdrant de leitura. Nenhum secret novo é necessário no GitHub.
A política nova lê os dados e grava apenas `mcq/current/*` no bucket de benchmarks.
Deploy, ETL e MCQ compartilham a concorrência `aws-production`. A parada de
emergência é verificada antes de iniciar e entre questões; o cancelamento do
workflow também interrompe a execução.
