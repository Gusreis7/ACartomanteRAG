# AWS: configuração inicial e operação pelo GitHub

O ETL é executado integralmente no Glue: crawler → Raw → Bronze → Silver → índice textual → enrichment → índice auxiliar. Os artefatos são gravados no S3 e os vetores no Qdrant Cloud. Não há migração de dados do notebook ou do Raspberry nesse fluxo.

## O que configurar, onde e quando

| Informação | Onde | Quando |
|---|---|---|
| Sessão AWS do usuário `terraform` | Perfis locais `terraform-login` e `terraform` já configurados | Bootstrap inicial e operações administrativas locais; renovar com `aws login`. |
| Região, domínio, e-mails, tamanho da EC2, volume e ID numérico do GitHub | `infra/aws/production.tfvars.json`, versionado | Antes do primeiro plano; depois por alteração de código. Não colocar segredos nesse arquivo. |
| Endpoint HTTPS e chaves Qdrant de produção | `config/aws-bootstrap.env`, privado e ignorado pelo Git | Antes do bootstrap; atualizar somente quando endpoint/chaves mudarem. |
| Políticas do grupo `server` | IAM da conta AWS | Uma vez antes do primeiro apply deste patch, incluindo as duas novas roles GitHub. |
| Roles OIDC e identificadores públicos de bootstrap | GitHub → Settings → Secrets and variables → Actions → Variables | O bootstrap cadastra automaticamente. Não cadastrar chaves AWS ou Qdrant em GitHub Secrets. |
| Segredos em execução | SSM Parameter Store, `SecureString` | O bootstrap importa do arquivo privado; EC2/Glue leem usando suas roles. |
| Delegação DNS do subdomínio | Provedor DNS de `gustavodreis.dev.br` | Uma vez após obter os nameservers, antes de publicar HTTPS. |
| Liberação do deploy da API | Variável GitHub `AWS_APP_DEPLOY_ENABLED=true` | Após o ETL promovido e a delegação DNS. |
| Versão do corpus e promoção | GitHub Actions → AWS ETL → Run workflow | Primeiro processamento e mudanças de dados/pipeline; não em cada feature. |

## 1. Preparar no notebook Debian

O Raspberry continua com os checks estáticos e o runtime suspenso. Gitea hospeda o repositório e espelha o GitHub; os runners da AWS são hospedados pelo GitHub.

São necessários AWS CLI v2.32+, Terraform **1.11.4**, GitHub CLI e Python **3.11+**. Docker no notebook não é necessário para este bootstrap: as imagens e wheels serão construídas no runner GitHub. O build real e a compatibilidade do Glue ainda devem ser confirmados na conta.

Na raiz do checkout:

```bash
export AWS_PROFILE=terraform
export AWS_REGION=us-east-1
aws sts get-caller-identity
gh auth status

test -d .venv || python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-runtime.lock boto3 python-dotenv
```

O ARN deve ser do usuário `terraform`. A operação de bootstrap recusa root. Para renovar a sessão, executar `aws login --profile terraform-login`; o perfil `terraform` continua usando o `credential_process` já configurado.

## 2. Preencher a configuração pública e o arquivo privado

Os comandos não sobrescrevem arquivos já existentes:

```bash
test -f infra/aws/production.tfvars.json || cp infra/aws/production.tfvars.json.example infra/aws/production.tfvars.json
test -f config/aws-bootstrap.env || cp config/aws-bootstrap.env.example config/aws-bootstrap.env
chmod 600 config/aws-bootstrap.env
```

Definir o repositório real e consultar o ID, sem mudar configurações:

```bash
github_repo="SEU_USUARIO/ACartomanteRAG"
gh api "repos/$github_repo" --jq '.id'
```

Editar `production.tfvars.json`: preencher `acme_email`, `budget_email` e `github_repository_id`. Conferir domínio, região, zona de disponibilidade, EC2, disco e orçamento. O padrão é `m7i-flex.large` (2 vCPUs, 8 GiB de RAM, x86_64), 40 GB de sistema e 50 GB de dados. Esse tipo foi confirmado como elegível ao Free Tier nesta conta em `us-east-1`; elegibilidade não elimina o consumo de créditos. O orçamento de US$ 75 é um alerta; não limita gastos. E-mails e IDs nesse arquivo serão versionados, sem credenciais.

Se o primeiro apply falhar por tipo de instância não elegível, alterar `instance_type` no arquivo preenchido para `m7i-flex.large`, preservando os demais campos, inclusive a AMI já fixada. Gerar um novo `bootstrap_aws.py plan`, revisar e então executar `apply`: o novo plano usa o state para reutilizar os recursos já criados. A configuração preenchida tem precedência sobre o default de `variables.tf`.

Se já há infraestrutura, transportar para esse arquivo as escolhas reais do antigo `terraform.tfvars`, inclusive `hosted_zone_id` e `oidc_provider_arn` quando usados. A configuração explícita de produção passa a ser a referência do bootstrap e do GitHub. O bootstrap fixa `ami_id` se estiver nulo, preservando a AMI do state existente ou consultando a AMI inicial.

Após transportar as escolhas, guardar o antigo `terraform.tfvars`, `terraform.tfvars.json` e eventuais `*.auto.tfvars*` fora do checkout. O bootstrap recusa esses arquivos carregados automaticamente para evitar uma configuração local diferente da usada pelo runner GitHub.

Em `config/aws-bootstrap.env`, preencher somente:

```dotenv
QDRANT_URL=https://SEU_CLUSTER_QDRANT:6333
QDRANT_READ_API_KEY=CHAVE_DE_LEITURA
QDRANT_WRITE_API_KEY=CHAVE_DE_ESCRITA
```

Usar o endpoint do Qdrant Cloud, sem credenciais embutidas na URL. O arquivo é lido como dotenv com interpolação desativada; não é executado com `source`. Outros campos não são importados. Chaves e senhas não entram no pacote da aplicação, no bundle Glue ou nos argumentos Terraform. As senhas Redis são geradas na primeira configuração e preservadas nas seguintes.

## 3. Atualizar a política IAM do grupo server

A política complementar anterior autorizava somente as roles de runtime, Glue e deploy. Agora também são necessárias `acartomanterag-github-infra` e `acartomanterag-github-etl`.

Gerar o JSON para a conta e o projeto configurados:

```bash
python scripts/bootstrap_aws.py policy > infra/aws/server-terraform-policy.local.json
```

Com um acesso administrativo que possa editar IAM, abrir **IAM → Policies → ServerTerraformIAM → Edit → JSON** e substituir pelo conteúdo gerado. Manter `PowerUserAccess` anexada ao grupo `server` e o usuário `terraform` nesse grupo. O gerador apenas consulta a identidade e imprime a política; não altera IAM.

O perfil local e a role GitHub de infraestrutura têm poder amplo de provisionamento, inclusive para modificar as políticas das roles do projeto. São acessos administrativos da infraestrutura e devem ficar restritos ao mantenedor e ao código confiável da `main`. A role do ETL aciona apenas o job e a recarga de dados; a de deploy publica a aplicação.

## 4. Gerar o plano e aplicar a primeira infraestrutura

```bash
python scripts/bootstrap_aws.py plan --repo "$github_repo"
```

Antes de provisionar, o script valida o arquivo privado e confirma que o ID do repositório corresponde ao arquivo público. Prepara o bucket de state exclusivo, privado, versionado e criptografado; inicializa o backend com lock no S3; valida Terraform e salva `infra/aws/bootstrap.tfplan`.

Conferir as alterações e os recursos com cobrança, especialmente EC2, discos, IP público, DNS e Glue. O plano não é enviado como artifact ao GitHub. Depois da revisão:

```bash
python scripts/bootstrap_aws.py apply --repo "$github_repo"
```

Esse comando aplica **o plano salvo**, sem outra confirmação interativa. Recusa mudanças de conta, repositório ou arquivos de infraestrutura após o plano; nesse caso, gerar `plan` novamente. Terraform também verifica a validade do state do plano.

Após o apply, o bootstrap grava os segredos no SSM e configura OIDC/variáveis do GitHub. Nenhum ETL é iniciado. Se a infraestrutura foi criada mas a configuração final falhou, corrigir a causa e repetir somente:

```bash
python scripts/bootstrap_aws.py configure --repo "$github_repo"
```

O bucket e a chave de state são derivados de projeto/conta/região. O formato padrão mantém compatibilidade com `acartomanterag-tfstate-<CONTA>-us-east-1` e `acartomanterag/production.tfstate`. O cadastro inicial do usuário/grupo e o próprio backend precedem o Terraform da aplicação.

## 5. Versionar e conferir o espelhamento

Versionar as alterações do patch, `production.tfvars.json` com a AMI resolvida e `infra/aws/.terraform.lock.hcl`. Não versionar `config/aws-bootstrap.env`, state, planos, `backend.hcl`, outputs ou arquivos `*.local.json`. Guardar a embalagem do ZIP fora do checkout.

Seguir o fluxo de branches do Gitea e conferir que a mesma revisão chegou à `main` do GitHub. Inicialmente `AWS_APP_DEPLOY_ENABLED` é criada como `false`; as validações e o Terraform podem rodar sem tentar iniciar uma API ainda sem corpus.

Por padrão o bootstrap usa o modo de repositório espelhado: configura OIDC e preserva as proteções GitHub existentes, sem criar a exigência de PR no GitHub que bloquearia o push espelhado. A proteção/revisão da `main` fica no Gitea. Se já existir no GitHub uma proteção incompatível com o espelhamento, ajustá-la conscientemente na configuração do repositório; o script não remove proteções existentes. Para um GitHub que seja a origem principal, `--protect-main` aplica a proteção de PR e check `validate`.

As variáveis cadastradas automaticamente incluem `AWS_REGION`, `AWS_INFRA_ROLE_ARN`, `AWS_ETL_ROLE_ARN`, `AWS_DEPLOY_ROLE_ARN`, `TF_STATE_BUCKET`, `TF_STATE_KEY`, `OPS_BUCKET` e `ECR_REPOSITORY`. IDs de instância/documentos ficam também nos outputs para diagnóstico, mas os workflows leem o contexto atualizado em `s3://<OPS_BUCKET>/config/deployment.json`. Um novo apply atualiza esse contexto, sem exigir recadastro manual do ID EC2.

## 6. Executar o ETL na nuvem pelo GitHub

Abrir **GitHub → Actions → AWS ETL → Run workflow**, sempre na branch `main`.

Primeiro smoke:

| Campo | Valor |
|---|---|
| operation | `smoke` |
| run_id | `smoke-v1` |
| revision | Vazio para usar a revisão exibida na execução |
| promote | Desmarcado |

O runner publica/reutiliza um bundle completo da revisão. O processamento ocorre no Glue, que testa bibliotecas, S3, Qdrant, catálogo Bedrock e um embedding Colibri. Smoke tem custo de Glue; não gera uma resposta LLM e não comprova permissão efetiva de inferência.

Depois, executar o pipeline completo:

| Campo | Valor |
|---|---|
| operation | `run` |
| run_id | `corpus-v1`, ou outro identificador novo |
| revision | Vazio no primeiro processamento; anotar o SHA mostrado |
| promote | Desmarcado para gerar um candidato; marcado para validar e publicar ao concluir |

Todas as seis etapas rodam na nuvem. O enriquecimento usa Qwen no Bedrock e tem custo de inferência. Não usa um índice local nem precisa de dados do Raspberry.

Para retomar ou promover um candidato, usar **o mesmo run_id e o mesmo SHA completo** da execução original. Na promoção, selecionar `run` e marcar `promote`: as etapas concluídas são reutilizadas e a promoção valida os artefatos antes de selecionar `production/current.json`. A inicialização de um novo job Glue ainda tem custo, mesmo reutilizando o trabalho concluído. Uma nova configuração/revisão exige novo run_id.

O workflow aceita somente revisões que já pertençam à história da `main` selecionada. Script, fontes e wheels são fixados por execução, usando os overrides de argumentos do Glue; não é necessário mudar `glue_bundle_prefix` e reaplicar Terraform a cada revisão. Um manifest final no S3 identifica um bundle completo. Artefatos existentes não são sobrescritos; publicação incompleta pode ser retomada.

Após a promoção, um documento SSM restrito recarrega a aplicação se houver release instalada. Se ainda não houve primeiro deploy, registra essa situação e não cria uma publicação incompleta. Cancelar o workflow não equivale a parar o Glue: conferir o job antes de repetir. Falhas no controlador após promoção podem ser recuperadas repetindo a mesma revisão/run_id. O lock de ETL e os checkpoints ficam no S3.

## 7. DNS e primeira publicação da aplicação

Consultar os nameservers com `terraform -chdir=infra/aws output nameservers`. No provedor do domínio, delegar somente `acartomanterag.gustavodreis.dev.br`, preservando os outros registros. Se já usa uma zona Route 53 existente configurada no Terraform, conferir o registro nessa zona.

Após o ETL promovido, a resolução DNS correta e a revisão do resultado:

1. Abrir **Settings → Secrets and variables → Actions → Variables** e mudar `AWS_APP_DEPLOY_ENABLED` para `true`.
2. Abrir **Actions → CI and AWS deploy → Run workflow → main**.
3. Aguardar os checks, Terraform, publicação da imagem e deploy via SSM. Conferir HTTPS, prontidão, uma pergunta real, reinício, restauração e rollback antes de considerar a publicação validada.

## Operação após o bootstrap

| Alteração | Fluxo |
|---|---|
| Feature de UI/API | Push/PR habitual → espelho GitHub → testes → Terraform plan/apply → imagem ECR e deploy. Não executa ETL. |
| Infraestrutura | Alterar `.tf` ou `production.tfvars.json`, revisar e publicar na main; o mesmo fluxo aplica antes do deploy da API. |
| Consultar um plano sem aplicar | Actions → AWS infrastructure → Run workflow → `plan`. |
| Aplicar infraestrutura separadamente | AWS infrastructure → `apply`; usa o plano gerado naquela execução. |
| Novos dados ou mudança no pipeline | AWS ETL com novo run_id e revisão escolhida. |
| Retomar/promover dados | AWS ETL com o run_id e SHA originais. |
| Rotacionar chave Qdrant | Atualizar `config/aws-bootstrap.env`, executar `bootstrap_aws.py configure` e o workflow de publicação da API para reler o SSM. Não reenriquecer o corpus. |

Terraform, ETL e deploy compartilham o grupo de concorrência `aws-production`, sem cancelar a execução ativa. O deploy da API aguarda o apply correspondente. O ETL continua sendo um acionamento explícito pelo workflow, para não reprocessar o corpus em cada feature.

O bootstrap é uma operação inicial assistida; as próximas publicações não precisam de Docker, arquivos privados ou login AWS do notebook. A conta conserva o state e os segredos, e o GitHub assume as roles temporariamente. Se a infraestrutura for recriada em outra conta, repetir o bootstrap com sua identidade, configuração pública e segredos daquele ambiente.

Referências: [variáveis Terraform](https://developer.hashicorp.com/terraform/language/values/variables), [state S3](https://developer.hashicorp.com/terraform/language/backend/s3), [OIDC GitHub/AWS](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws), [argumentos por execução Glue](https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-etl-glue-arguments.html).
