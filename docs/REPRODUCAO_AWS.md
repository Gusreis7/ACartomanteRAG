# Publicar em uma conta AWS própria

O caminho é: preparar a conta, aplicar a infraestrutura, publicar os dados e liberar a aplicação. São necessários notebook, repositório GitHub, domínio e Qdrant Cloud. **Raspberry e Gitea não são necessários.** Eu os uso no homelab; uma reprodução pode trabalhar diretamente no GitHub.

Os jobs usam serviços reais e consomem créditos ou geram cobrança conforme a conta. O orçamento configurado é um alerta, não um teto de gastos.

## 1. Preparar conta e configuração

Crie seu fork no GitHub, clone-o e entre na raiz do projeto. Instale AWS CLI v2, Terraform 1.11.4, GitHub CLI e Python 3.11+. Configure um perfil AWS administrativo para o bootstrap, sem usar root. No meu ambiente, ele se chama `terraform`, pertence ao grupo `server` e usa `PowerUserAccess` mais a política complementar `ServerTerraformIAM`. Esse acesso fica no notebook.

```bash
python3 -m venv .venv
source .venv/bin/activate
python scripts/dependencies.py install ops
export AWS_PROFILE=terraform
export AWS_REGION=us-east-1
github_repo='SEU_USUARIO/SEU_REPOSITORIO'
aws sts get-caller-identity
gh auth status
test -f infra/aws/production.tfvars.json || cp infra/aws/production.tfvars.json.example infra/aws/production.tfvars.json
test -f config/aws-bootstrap.env || cp config/aws-bootstrap.env.example config/aws-bootstrap.env
chmod 600 infra/aws/production.tfvars.json config/aws-bootstrap.env
gh api "repos/$github_repo" --jq '.id'
```

| Onde configurar | O que preencher |
| --- | --- |
| `infra/aws/production.tfvars.json` | Sua região, domínio, e-mails, capacidade e `github_repository_id` retornado acima. É o ID do repositório, não do proprietário. |
| `config/aws-bootstrap.env` | `QDRANT_URL`, `QDRANT_READ_API_KEY` e `QDRANT_WRITE_API_KEY` do seu cluster; separe leitura e escrita. |
| Perfil AWS local | Autenticação AWS. Não coloque access keys no `.env` ou nos tfvars. |

Confira a elegibilidade de `m7i-flex.large` no plano da sua conta. Se mudar a região, revise também `generation.region` e `enrichment.region` em `config/config.aws.yaml` e a disponibilidade dos modelos. O primeiro plano preenche `ami_id`; preserve esse valor depois.

Gere a política complementar e instale-a com um administrador no IAM:

```bash
python scripts/bootstrap_aws.py policy > infra/aws/server-terraform-policy.local.json
```

O `.env` do Compose local não configura esta publicação. O bootstrap lê seu arquivo privado, grava segredos no SSM e usa o perfil AWS pelo mecanismo padrão de credenciais.

## 2. Aplicar a infraestrutura

```bash
python scripts/bootstrap_aws.py plan --repo "$github_repo"
```

Essa preparação já pode criar o bucket de state. Revise recursos, capacidade e custos do plano salvo; depois:

```bash
python scripts/bootstrap_aws.py apply --repo "$github_repo"
```

O apply executa o plano salvo. Ao final, o bootstrap configura SSM, confiança OIDC, Variables e o Secret `TF_PRODUCTION_TFVARS_JSON` no seu GitHub. Na primeira instalação, `AWS_APP_DEPLOY_ENABLED=false` segura a API até os dados estarem prontos. Senhas Redis existentes são preservadas.

**Não é preciso cadastrar chaves AWS fixas no GitHub.** OIDC emite sessões temporárias para o ID do repositório e a `main`. O Secret dos tfvars guarda configuração privada; as chaves Qdrant vão para SSM. ARNs e identificadores ficam em Variables.

Se só a configuração final falhar, use `python scripts/bootstrap_aws.py configure --repo "$github_repo"`. Se o apply falhar parcialmente, corrija a causa e gere outro plano sobre o mesmo state. Versione código, exemplos e `.terraform.lock.hcl`; mantenha configuração real, backend gerado, state e planos fora do Git.

Quando alterar os tfvars privados, atualize a cópia da CI, já com a AMI fixada:

```bash
gh secret set TF_PRODUCTION_TFVARS_JSON --repo "$github_repo" < infra/aws/production.tfvars.json
```

### Permissões da automação

A role infra do GitHub usa leitura e operações específicas nos recursos existentes. IAM, rede, proteção de buckets, novos recursos e substituições exigem plano/apply administrativo no notebook. As políticas ficam em `infra/aws/infra_permissions.tf`; `scripts/check_infra_plan.py` confere o escopo do plano na CI.

Em instalação antiga, aplique localmente a transição que remove `github_infra_poweruser` e instala as duas políticas inline. **O teste aprovado não altera a conta.** Depois do apply, confira:

```bash
aws iam list-attached-role-policies --role-name acartomanterag-github-infra \
  --query 'AttachedPolicies[].PolicyName' --output json --no-cli-pager
aws iam list-role-policies --role-name acartomanterag-github-infra \
  --query PolicyNames --output json --no-cli-pager
```

O esperado é `[]` nos attachments e duas políticas inline. Para testar a barreira de autoalteração:

```bash
ROLE_ARN="$(terraform -chdir=infra/aws output -raw github_infra_role_arn)"
aws iam simulate-principal-policy --policy-source-arn "$ROLE_ARN" \
  --action-names iam:PutRolePolicy iam:AttachRolePolicy iam:UpdateAssumeRolePolicy \
  --resource-arns "$ROLE_ARN" \
  --query 'EvaluationResults[].{Acao:EvalActionName,Decisao:EvalDecision}' \
  --output table --no-cli-pager
```

As três devem retornar `explicitDeny`. Publique o código e execute **AWS infrastructure → plan** para conferir também o acesso real da pipeline.

## 3. Coletar somente o Raw no notebook

O MEC respondeu 403 ao Glue e 200 ao notebook. Deixei só o crawler local, gravando diretamente no S3; o restante segue na nuvem. A decisão está em [Impedimentos](IMPEDITIVOS.md#mec-coleta-fora-do-glue).

```bash
python scripts/dependencies.py install crawler
RAW_BUCKET="$(terraform -chdir=infra/aws output -json buckets \
  | python -c 'import json,sys; print(json.load(sys.stdin)["RAW_BUCKET"])')"
RAW_ID='machado-v1'
python -m src.etl.crawler_ingestao_raw_rustfs \
  --raw-bucket "$RAW_BUCKET" --raw-id "$RAW_ID" --region "$AWS_REGION" \
  --pagina-inicial 1 --pagina-final 7 --delay 2 --strict
```

Mantenha o notebook conectado até concluir. Para tentar apenas pendências:

```bash
python -m src.etl.crawler_ingestao_raw_rustfs \
  --raw-bucket "$RAW_BUCKET" --raw-id "$RAW_ID" --region "$AWS_REGION" --retry
```

Se confirmar manualmente um PDF indisponível na fonte, registre seu ID com `--mark-unavailable ID --reason "Motivo da conferência"`. Na minha coleta foram `000082`, `000122` e `000197`; não copie essas exclusões sem conferir sua própria coleta.

Código 2 indica pendências salvas. Só continue após `Raw pronto para o Glue` e confira o recibo:

```bash
aws s3 cp "s3://$RAW_BUCKET/production/raw/$RAW_ID/ready.json" - --only-show-errors
```

O snapshot pronto é imutável. Outra consulta usa outro `raw_id`.

## 4. Processar e promover os dados

Com a revisão na `main`, abra **Actions → AWS ETL → Run workflow**. Faça primeiro `operation=smoke`, com um ID de teste, demais IDs vazios e promoção desmarcada. Esse teste usa dependências e inferência reais.

Depois do smoke e do Raw pronto:

| Campo | Primeira execução completa |
| --- | --- |
| `operation` | `run-from-raw` |
| `raw_id` | O snapshot coletado, por exemplo `machado-v1` |
| `run_id` | Novo ID, por exemplo `corpus-raw-v1` |
| `revision` | Vazio usa o SHA da execução; anote-o completo |
| `resume_from_revision` | Vazio |
| `promote` | Desmarcado |

Glue prepara Bronze, Silver, embeddings, enrichment e índice auxiliar. Confira o candidato e os manifests; repita os mesmos parâmetros com `promote=true` para publicar a release. Isso solicita recarga de uma API existente, mas não faz a primeira publicação da aplicação.

### Retomar e diagnosticar o ETL

Na retomada comum, preserve `raw_id`, `run_id` e SHA, com promoção desmarcada. Etapas concluídas são reutilizadas; o manifest retém pendências. O `jr_...` identifica uma tentativa Glue, não o checkpoint do corpus.

No Console: **Glue → ETL jobs → acartomanterag-etl → Runs**. Logs no CloudWatch: `/aws-glue/jobs/output` e `/aws-glue/jobs/error`.

```bash
GLUE_RUN='jr_ID_DA_EXECUCAO'
aws glue get-job-run --job-name acartomanterag-etl --run-id "$GLUE_RUN" \
  --query 'JobRun.{Estado:JobRunState,Erro:ErrorMessage}' --output json --no-cli-pager
python -u scripts/cloud.py wait-glue --job-run-id "$GLUE_RUN" --wait-seconds 8100
```

Não libere `production/etl.lock` sem confirmar que o produtor encerrou. Se precisar migrar código, `resume_from_revision` recebe a revisão efetiva do checkpoint, inclusive migrações anteriores. O ETL valida assinatura e configuração; editar hashes à mão só torna o problema mais criativo.

Respostas rejeitadas pelo enrichment ficam no objeto apontado por `response_error_bucket`/`response_error_key`, também registrado em `erro_detalhes_json`. Consulte o diagnóstico antes de atribuir tudo à quota. Retry, timeout e limitações estão em [Impedimentos](IMPEDITIVOS.md).

## 5. Configurar DNS e publicar a API

Consulte os outputs `public_ip` e `nameservers` com `terraform -chdir=infra/aws output`. Escolha um caminho:

- **Provedor DNS atual:** crie o A do subdomínio na zona autoritativa, apontando para o EIP. Foi o que usei no Registro.br.
- **Route 53:** delegue somente o subdomínio com registros NS no domínio pai, usando os nameservers do output. O A será servido pela zona AWS.

A zona Route 53 criada não assume o DNS sozinha. Não mantenha um A concorrente no nome delegado. Confira a resolução com `dig +short A SEU_SUBDOMINIO`.

Com DNS correto e dados promovidos:

```bash
gh variable set AWS_APP_DEPLOY_ENABLED --body true --repo "$github_repo"
make aws GITHUB_REPO="$github_repo"
make aws-status GITHUB_REPO="$github_repo"
```

Push na `main` também dispara CI/CD. Se a revisão já está executando, acompanhe em vez de disparar outra. Abra a URL HTTPS e teste pergunta literária, recusa, isolamento entre sessões e novo chat. `/health` e `/ready` são internos na AWS.

Valide também reinício, restauração de um backup conhecido e rollback. Os comandos de operação estão na [Infra atual](INFRA_ATUAL.md#operar-e-recuperar). Novas features seguem CI/CD; mudanças de dados exigem promoção própria, e mudanças administrativas continuam no perfil local. Nos atalhos Make, informe sempre seu `GITHUB_REPO`: o padrão do projeto aponta para o meu.
