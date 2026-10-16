# Coleta Raw no notebook e processamento no Glue

## Contexto e decisão

Em 15/10/2026, o crawler no Glue recebeu HTTP 403 nas sete páginas do Domínio Público/MEC. A mesma consulta no notebook retornou HTTP 200. A causa exata da diferença não foi identificada.

A coleta passou a ser executada uma única vez no notebook, com envio direto dos PDFs e do manifest ao S3 Raw. Esse snapshot permanece como entrada dos processamentos seguintes, sem nova consulta ao portal.

| Etapa | Execução | Resultado |
|---|---|---|
| Crawler | Notebook | PDFs e manifest no S3 Raw |
| Bronze e Silver | Glue | Textos e chunks fixed/256/o32 no S3 |
| Índice textual | Glue | Embeddings Colibri no Qdrant Cloud |
| Enrichment | Glue + Bedrock | Resumos Qwen3 32B no S3 |
| Índice auxiliar | Glue | Embeddings dos resumos no Qdrant Cloud |

## 1. Coletar o corpus

Pré-requisitos: bucket Raw criado pelo Terraform, perfil `terraform` configurado e código com suporte a `--raw-id`. A mesma revisão deve estar na `main` do GitHub antes da execução no Glue.

Na raiz do projeto:

```bash
source .venv/bin/activate
python -m pip install -c requirements-glue.lock -r requirements-crawler.txt
export AWS_PROFILE=terraform
export AWS_REGION=us-east-1
aws sts get-caller-identity

python -m src.etl.crawler_ingestao_raw_rustfs \
  --raw-bucket acartomanterag-681835981654-us-east-1-raw \
  --raw-id machado-v1 \
  --region us-east-1 \
  --pagina-inicial 1 --pagina-final 7 \
  --delay 2 --strict
```

Se a sessão expirar, renove com `aws login --profile terraform-login`. O crawler usa o perfil AWS, sem carregar o `.env` dos serviços locais. Cada PDF passa por um arquivo temporário antes do envio; o notebook deve permanecer conectado até concluir.

## 2. Confirmar a conclusão

Os arquivos ficam em `production/raw/machado-v1/`: `request.json` registra a consulta; `documents/` contém os PDFs; `manifest.parquet` mantém metadados e checkpoints; `ready.json` confirma a conclusão.

O recibo só é publicado após conferir páginas e integridade dos PDFs. Uma obra com falha permanece pendente até ser recuperada ou ter sua indisponibilidade confirmada por ID e motivo. Essas exclusões ficam no manifest e no recibo; a Bronze usa apenas PDFs válidos.

Após a mensagem `Raw pronto para o Glue`, consulte:

```bash
aws s3 cp \
  s3://acartomanterag-681835981654-us-east-1-raw/production/raw/machado-v1/ready.json \
  - --region us-east-1
```

## 3. Processar no Glue

Abra **GitHub → Actions → AWS ETL → Run workflow → main**:

| Campo | Valor |
|---|---|
| `operation` | `run-from-raw` |
| `raw_id` | `machado-v1` |
| `run_id` | `corpus-raw-v1`, novo para este processamento |
| `revision` | Vazio inicialmente; anote o SHA completo exibido |
| `promote` | Desmarcado para avaliar o candidato |

O Glue verifica o Raw e executa Bronze, Silver, índice textual, enrichment e índice auxiliar. O modo `run` inclui o crawler na nuvem e pode reencontrar o 403; o fluxo adotado é `run-from-raw`.

O processamento permanece em CPU no plano FREE. Após aplicar a atualização Terraform, o prazo do Glue é de dez horas; o GitHub acompanha o mesmo job em janelas com renovação de OIDC. O procedimento para ativar esse limite e retomar a revisão existente está em [AWS_AUTOMATION.md](AWS_AUTOMATION.md#execuções-longas-em-cpu).

## 4. Retomar e promover

- **Downloads pendentes:** use `--retry` com o mesmo bucket e `raw_id`. Só as obras pendentes são revisitadas. O manifest registra status, erro, contador de tentativas desta versão e data da última tentativa.
- **Glue interrompido:** repita `run-from-raw` com os mesmos `raw_id`, `run_id` e SHA. O enrichment retoma somente pendências. Atualizações operacionais do Mantle admitem [adoção explícita da nova revisão](AWS_AUTOMATION.md#pendências-do-enrichment), preservando os resultados salvos.
- **Promoção:** repita esses mesmos parâmetros e marque `promote`. O job valida os índices, atualiza `production/current.json` e solicita a recarga da API existente.

Uma interrupção abrupta pode deixar `production/raw/machado-v1/crawl.lock`. Remova somente esse lock após confirmar que a coleta encerrou. Preserve os objetos de um Raw concluído para manter a integridade e a retomada.

```bash
python -m src.etl.crawler_ingestao_raw_rustfs \
  --raw-bucket acartomanterag-681835981654-us-east-1-raw \
  --raw-id machado-v1 --retry
```

Para as três obras conferidas manualmente como indisponíveis:

```bash
python -m src.etl.crawler_ingestao_raw_rustfs \
  --raw-bucket acartomanterag-681835981654-us-east-1-raw \
  --raw-id machado-v1 \
  --mark-unavailable 000082 000122 000197 \
  --reason "PDFs indisponíveis na fonte; conferência manual"
```

Os erros originais são preservados junto do motivo e da data de confirmação. Os 318 PDFs disponíveis podem seguir para o Glue após a validação. Publique esta revisão no GitHub e use seu SHA na execução: o recibo com exclusões é versão 2 e exige o leitor atualizado.

`crawl-status.json` guarda a conferência das páginas. Na primeira retomada de um manifest antigo, apenas as listagens são reconsultadas; os PDFs concluídos ficam preservados. Nas seguintes, a conferência é reutilizada. Pendências encerram com resumo e código 2, sem traceback; Raw pronto retorna código 0. Outra consulta ou intervalo exige novo `raw_id`.

## Limites

O snapshot representa os registros retornados pela consulta e pelas páginas selecionadas. A coleta inicial depende do acesso do notebook ao portal. As verificações releem os PDFs no S3; Glue, armazenamento e inferência mantêm seus custos. Uma chamada LLM interrompida antes do checkpoint pode se repetir.

DNS, deploy da API e operação posterior: [AWS_AUTOMATION.md](AWS_AUTOMATION.md).
