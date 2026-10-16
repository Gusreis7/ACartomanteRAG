# Infraestrutura AWS e publicação da aplicação

## Arquitetura e decisões

O Gitea no Raspberry mantém o repositório e o espelho do GitHub. O GitHub Actions executa CI/CD; o runtime local permanece suspenso durante a manutenção do Raspberry.

O Terraform define rede, EC2, discos, buckets, ECR, IAM, SSM, Glue, DNS e alerta de orçamento. O state fica no S3 com lock. A coleta Raw ocorre uma vez no notebook; as demais etapas do ETL rodam no Glue, com vetores no Qdrant Cloud.

| Configuração | Local | Uso |
|---|---|---|
| Região, domínio, e-mails, ID do GitHub, EC2, discos e AMI | `infra/aws/production.tfvars.json` | Configuração pública versionada |
| Sessão do usuário `terraform` | Perfis locais `terraform-login` e `terraform` | Bootstrap, coleta e administração |
| Endpoint e chaves Qdrant | `config/aws-bootstrap.env` | Arquivo privado importado no bootstrap |
| Segredos de execução | SSM Parameter Store | EC2 e Glue acessam pelas próprias roles |
| Roles e identificadores públicos | GitHub Actions → Variables | Cadastrados pelo bootstrap |

O GitHub usa OIDC para obter credenciais AWS temporárias; as chaves Qdrant ficam no SSM. As roles de infraestrutura, deploy e ETL têm funções separadas. O perfil local e a role de infraestrutura possuem acesso administrativo amplo; a revisão da `main` no Gitea controla o código publicado pelo espelho.

## 1. Preparar um ambiente novo

Ferramentas: AWS CLI v2.32+, Terraform 1.11.4, GitHub CLI e Python 3.11+. Na raiz do projeto:

```bash
export AWS_PROFILE=terraform
export AWS_REGION=us-east-1
aws sts get-caller-identity
gh auth status

test -d .venv || python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-runtime.lock boto3 python-dotenv
```

Confirme a identidade `terraform`. Para renovar a sessão: `aws login --profile terraform-login`. Em novos terminais, selecione novamente `AWS_PROFILE=terraform`.

## 2. Configurar o bootstrap

Crie os arquivos apenas se ainda não existirem:

```bash
test -f infra/aws/production.tfvars.json || cp infra/aws/production.tfvars.json.example infra/aws/production.tfvars.json
test -f config/aws-bootstrap.env || cp config/aws-bootstrap.env.example config/aws-bootstrap.env
chmod 600 config/aws-bootstrap.env

github_repo="Gusreis7/ACartomanteRAG"
gh api "repos/$github_repo" --jq '.id'
```

Preencha o arquivo público com os valores da tabela e o ID retornado. Em infraestrutura existente, preserve a AMI, as escolhas do state e os IDs de zona/OIDC usados. O bootstrap fixa a AMI inicial quando necessário; versionar esse valor evita trocas involuntárias. Ele recusa arquivos `terraform.tfvars` ou `*.auto.tfvars*` concorrentes no checkout.

No arquivo privado, preencha:

```dotenv
QDRANT_URL=https://SEU_CLUSTER_QDRANT:6333
QDRANT_READ_API_KEY=CHAVE_DE_LEITURA
QDRANT_WRITE_API_KEY=CHAVE_DE_ESCRITA
```

O bootstrap lê esse arquivo e grava os valores em SSM `SecureString`. As senhas Redis são geradas inicialmente e preservadas nas configurações seguintes.

Para preparar a política IAM complementar:

```bash
python scripts/bootstrap_aws.py policy > infra/aws/server-terraform-policy.local.json
```

Com acesso administrativo, atualize **IAM → Policies → ServerTerraformIAM** com o JSON gerado. O usuário `terraform` pertence ao grupo `server`, com essa política e `PowerUserAccess`.

## 3. Provisionar e publicar o código

Gere o plano:

```bash
python scripts/bootstrap_aws.py plan --repo "$github_repo"
```

Revise os recursos e os custos previstos. Depois, aplique o plano salvo:

```bash
python scripts/bootstrap_aws.py apply --repo "$github_repo"
```

O comando prepara o SSM e as variáveis/OIDC do GitHub após o apply. Se essa configuração final falhar, corrija a causa e execute `python scripts/bootstrap_aws.py configure --repo "$github_repo"`. Alterações de conta, repositório, infraestrutura ou configuração após o plano exigem novo `plan`.

Versione o código, `production.tfvars.json` e `infra/aws/.terraform.lock.hcl`; mantenha segredos, state, planos, `backend.hcl` e outputs locais fora do Git. Confira a mesma revisão na `main` do GitHub após o push ao Gitea. O deploy da API começa desabilitado por `AWS_APP_DEPLOY_ENABLED=false`.

## 4. Preparar os dados

Em **GitHub → Actions → AWS ETL → main**, rode `operation=smoke`, `run_id=smoke-v1`, revisão vazia e promoção desmarcada. O smoke verifica bibliotecas, S3, Qdrant, catálogo, uma inferência de até 64 tokens de saída e Colibri.

Siga [AWS_RAW_CRAWL.md](AWS_RAW_CRAWL.md) para coletar o Raw e executar `run-from-raw` no Glue. Para retomar ou promover, mantenha o mesmo `raw_id`, `run_id` e SHA. Os bundles são fixados por revisão, sem reaplicar Terraform para cada versão do ETL.

A promoção seleciona `production/current.json` e solicita a recarga da API. Antes do primeiro deploy, a recarga registra que a aplicação ainda não foi publicada. Cancelar o workflow não encerra necessariamente o Glue: confira o job antes de repetir.

### Execuções longas em CPU

O plano FREE foi mantido. O Glue tem limite de dez horas com os mesmos dois workers `G.1X`, sem repetição automática. O workflow inicia um único `jr_` e o acompanha em cinco janelas de até 135 minutos. Cada job renova o OIDC; os seguintes conferem o mesmo Glue, inclusive se ele já terminou. O resultado final exige `SUCCEEDED`. A concorrência `aws-production` cobre o workflow inteiro.

Para ativar o novo prazo, publique o patch no Gitea e confira o espelho da `main`. Execute **AWS infrastructure → plan**, revise o novo plano e depois **apply**; o ajuste esperado é `aws_glue_job.etl.timeout`, de 120 para 600. Outros desvios exigem revisão. Não reutilize um plano salvo antes da alteração. O novo workflow confere o prazo provisionado antes de iniciar outro ETL.

Uma execução já iniciada mantém seu próprio `StartedOn`, `Timeout` e bundle. Retomadas comuns preservam Raw, `run_id` e revisão ETL. Para trocar a rota Converse por Mantle no checkpoint `corpus-raw-v1`, use o procedimento abaixo.

O checkpoint fica no bucket de índices, em `production/runs/corpus-raw-v1.json`. `completed` registra etapas concluídas; `current_step` identifica a etapa atual. O horário muda nas transições, sem representar um sinal periódico de atividade. Os logs detalhados ficam em `/aws-glue/jobs/output` e `/aws-glue/jobs/error`, filtrados pelo `jr_`.

Se uma interrupção forçada deixar `production/etl.lock`, confirme no Glue que não existe execução em `STARTING`, `RUNNING`, `WAITING` ou `STOPPING`. Confira também que o conteúdo do lock corresponde ao `run_id` interrompido. Só então remova esse objeto específico no bucket de índices. Preserve manifest, candidatos, PDFs e collections. O índice parcial é recalculado na retomada; etapas completas e os checkpoints do enrichment são reaproveitados com a mesma revisão.

O encerramento de uma janela de acompanhamento não para o Glue. Também não há encerramento automático do job AWS ao cancelar o workflow. Retomar o acompanhamento de um `jr_` existente não exige iniciar outro ETL:

```bash
export AWS_PROFILE=terraform
export AWS_REGION=us-east-1
python -u scripts/cloud.py wait-glue --job-run-id jr_ID_DA_EXECUCAO --wait-seconds 8100
```

Esse comando usa os outputs do Terraform local; `--outputs` aceita o contexto JSON existente. Retorno 0 indica sucesso do Glue; 3 indica que ele ainda estava ativo no fim da janela. Falhas e timeout são propagados. As estimativas e a decisão financeira estão em [IMPEDITIVOS.md](IMPEDITIVOS.md).

### Qwen no Mantle

O overlay AWS usa `provider: bedrock-mantle` e `model: qwen.qwen3-32b`, com IAM/SigV4. Nenhum secret novo é necessário no GitHub ou SSM. O provider `bedrock` continua disponível para Converse, inclusive Scout; selecione provider e modelo juntos ao configurar esse benchmark.

1. Publique o patch no Gitea e confira o espelho na `main` do GitHub.
2. Execute **AWS infrastructure → plan** e depois **apply**, revisando as permissões Mantle das roles de aplicação e Glue. Não reaproveite plano antigo.
3. Execute **AWS ETL → smoke**, revisão vazia e sem promoção. Só continue após sucesso.
4. Execute **AWS ETL** com os campos abaixo para migrar o checkpoint já existente.

| Campo | Migração de `corpus-raw-v1` |
|---|---|
| `operation` / `raw_id` / `run_id` | `run-from-raw` / `machado-v1` / `corpus-raw-v1` |
| `revision` | Vazio; use a nova revisão exibida e anote seu SHA |
| `resume_from_revision` | `de3ac95e91500f26dcdf43212526fb52f33a2916` |
| `promote` | Desmarcado |

A migração aceita apenas as três primeiras etapas completas, Raw e configuração correspondentes à assinatura antiga, manifests consistentes e contagem exata no Qdrant. Salva o checkpoint anterior em `production/runs/corpus-raw-v1/before-mantle-<assinatura>.json`. Reutiliza o índice textual e inicia o enrichment da nova rota, sem alterar os artefatos antigos. Divergências interrompem a migração.

Nas retomadas e na promoção seguintes, mantenha o **novo SHA**, Raw e `run_id`; deixe `resume_from_revision` vazio. Confira que não existe Glue ativo antes de iniciar outra execução; um lock órfão segue o procedimento acima.

### Pendências do enrichment

`response_retries: 3` permite a chamada inicial e mais três tentativas para resposta vazia, truncada ou JSON inválido, com espera de 10/20/40 segundos. O manifest mantém status, `tentativas_enrichment`, consumo observado e `erro_tipo`/`erro_detalhes_json`. HTTP 429 e erros de permissão/serviço pausam o lote após o tratamento limitado de transporte; uma quota diária identificada não é repetida imediatamente.

O Glue usa `--retry` (alias de `--resume`): seleciona somente obras pendentes/erro e reutiliza checkpoints das unidades. Um lote incompleto salva o manifest e encerra com código 2, sem índice auxiliar ou promoção. Consulte `generation_response_error` e `bedrock_unavailable` nos logs: `finish_reason`, `max_tokens`, tokens de saída/raciocínio e request ID permitem investigar o corte. O erro antigo, sozinho, não comprova falta de quota.

Para adotar este patch em `corpus-raw-v1`, publique a nova revisão e use `revision` vazio, `resume_from_revision` igual ao SHA atualmente registrado no checkpoint, mesmo Raw/run e promoção desmarcada. A atualização verifica a assinatura e salva backup; preserva o dataset e os checkpoints Mantle quando só muda o código/controle de execução. Nas retomadas seguintes, fixe o novo SHA e deixe o campo de origem vazio. Nenhuma mudança adicional de IAM ou Terraform é necessária se a rota Mantle já foi provisionada.

## 5. Configurar DNS e publicar a API

Consulte os servidores da zona:

```bash
terraform -chdir=infra/aws output nameservers
```

Na zona autoritativa de `gustavodreis.dev.br`, delegue `acartomanterag.gustavodreis.dev.br` para esses servidores, preservando os demais registros. Se o Terraform utiliza uma zona existente, confira o registro nela.

Com DNS resolvendo e dados promovidos:

1. Defina `AWS_APP_DEPLOY_ENABLED=true` nas variáveis do GitHub Actions.
2. Execute **CI and AWS deploy → Run workflow → main**.
3. Valide HTTPS, prontidão, uma pergunta real, reinício, restauração e rollback.

## Operação posterior

| Alteração | Procedimento |
|---|---|
| Feature da aplicação | Gitea → espelho GitHub → testes → Terraform → imagem ECR → deploy SSM. O ETL é separado. |
| Infraestrutura | Atualizar `.tf`/configuração pública e publicar; ou usar **AWS infrastructure → plan/apply**. |
| Reprocessamento de dados | AWS ETL `run-from-raw`, mesmo Raw e novo `run_id` para outra revisão/configuração. |
| Rotação Qdrant | Atualizar o arquivo privado, executar `bootstrap_aws.py configure` e publicar novamente a API. |

Terraform, deploy e ETL compartilham a concorrência `aws-production`. Os workflows consultam o contexto atualizado no S3, sem depender de IDs EC2 antigos. O benchmark Qwen/Scout permanece separado do deploy e reutiliza a base, com DEV e holdout separados.
