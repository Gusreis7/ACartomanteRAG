# Infraestrutura AWS e publicação da aplicação

## Arquitetura e decisões

O Gitea no Raspberry mantém o repositório e o espelho do GitHub. O GitHub Actions executa CI/CD; o runtime local permanece suspenso durante a manutenção do Raspberry.

Branches de desenvolvimento e PRs são tratados no Gitea. No GitHub, `CI and AWS deploy` dispara somente por push na `main` ou manualmente; os testes, scanner, build e validação Terraform da revisão publicada continuam ativos. Isso elimina a segunda passagem `develop` → PR → `main` no espelho. O bootstrap em modo `--mirror-source` também deixa de criar `develop` no GitHub. As checagens locais do Gitea continuam sujeitas à limitação do Raspberry descrita em [IMPEDITIVOS.md](IMPEDITIVOS.md).

O Terraform define rede, EC2, discos, buckets, ECR, IAM, SSM, Glue, DNS e alerta de orçamento. O state fica no S3 com lock. A coleta Raw ocorre uma vez no notebook; as demais etapas do ETL rodam no Glue, com vetores no Qdrant Cloud.

| Configuração | Local | Uso |
|---|---|---|
| Região, domínio, e-mails, ID do GitHub, EC2, discos e AMI | `infra/aws/production.tfvars.json` | Arquivo privado, fora do Git |
| Cópia dos tfvars para a infraestrutura na CI | GitHub Actions → Secret `TF_PRODUCTION_TFVARS_JSON` | Runner reconstrói o JSON temporariamente |
| Sessão do usuário `terraform` | Perfis locais `terraform-login` e `terraform` | Bootstrap, coleta e administração |
| Endpoint e chaves Qdrant | `config/aws-bootstrap.env` | Arquivo privado importado no bootstrap |
| Segredos de execução | SSM Parameter Store | EC2 e Glue acessam pelas próprias roles |
| Roles e identificadores públicos | GitHub Actions → Variables | Cadastrados pelo bootstrap |

O GitHub usa OIDC para obter credenciais AWS temporárias; as chaves Qdrant ficam no SSM. O Secret dos tfvars guarda escolhas de infraestrutura, sem access keys. As roles de infraestrutura, deploy e ETL têm funções separadas. O perfil local e a role de infraestrutura possuem acesso administrativo amplo; a revisão da `main` no Gitea controla o código publicado pelo espelho.

## 1. Preparar um ambiente novo

Ferramentas: AWS CLI v2.32+, Terraform 1.11.4, GitHub CLI e Python 3.11+. Na raiz do projeto:

```bash
export AWS_PROFILE=terraform
export AWS_REGION=us-east-1
aws sts get-caller-identity
gh auth status

test -d .venv || python3 -m venv .venv
source .venv/bin/activate
python scripts/dependencies.py install ops
```

Confirme a identidade `terraform`. Para renovar a sessão: `aws login --profile terraform-login`. Em novos terminais, selecione novamente `AWS_PROFILE=terraform`.

## 2. Configurar o bootstrap

Crie os arquivos apenas se ainda não existirem:

```bash
test -f infra/aws/production.tfvars.json || cp infra/aws/production.tfvars.json.example infra/aws/production.tfvars.json
test -f config/aws-bootstrap.env || cp config/aws-bootstrap.env.example config/aws-bootstrap.env
chmod 600 infra/aws/production.tfvars.json config/aws-bootstrap.env

github_repo="Gusreis7/ACartomanteRAG"
gh api "repos/$github_repo" --jq '.id'
```

Preencha os tfvars com os valores da tabela e o ID retornado. Em infraestrutura existente, preserve a AMI, as escolhas do state e os IDs de zona/OIDC usados. O bootstrap fixa a AMI inicial quando necessário; conservar esse valor evita trocas involuntárias. Ele recusa arquivos `terraform.tfvars` ou `*.auto.tfvars*` concorrentes no checkout.

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

O comando prepara o SSM, as variáveis/OIDC e o Secret dos tfvars no GitHub após o apply. Se essa configuração final falhar, corrija a causa e execute `python scripts/bootstrap_aws.py configure --repo "$github_repo"`. Alterações de conta, repositório, infraestrutura ou configuração após o plano exigem novo `plan`.

Versione o código, o exemplo dos tfvars e `infra/aws/.terraform.lock.hcl`. Guarde os tfvars reais, segredos, state, planos, `backend.hcl` e outputs locais fora do Git. Conserve uma cópia privada do JSON para reproduzir o ambiente. Confira a mesma revisão na `main` do GitHub após o push ao Gitea. O deploy da API começa desabilitado por `AWS_APP_DEPLOY_ENABLED=false`.

Para uma infraestrutura já criada, cadastre ou atualize somente a cópia privada antes de publicar este ajuste:

```bash
gh secret set TF_PRODUCTION_TFVARS_JSON --repo "$github_repo" < infra/aws/production.tfvars.json
```

O workflow recebe esse Secret explicitamente, valida repositório/região/AMI e grava o JSON com permissão `600`. Ao terminar, remove o arquivo e o plano. Strings da configuração são mascaradas nos logs; identificadores e IPs gerados pelo Terraform ainda podem aparecer. O state continua privado no S3. Se o JSON já estiver versionado, use `git rm --cached -- infra/aws/production.tfvars.json` para parar de acompanhá-lo sem apagar a cópia local; commits antigos conservam o conteúdo.

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

Na run `jr_299f37f11d5012c96cd41e6ebbd3db6d15cf1e0ab6d8c284485a950b374a9323`, 253 das 257 obras ficaram prontas. O exemplo `finish_reason=length`, com 1.024 tokens gerados e permitidos, comprova o corte no teto solicitado; o log isolado não diagnostica as outras três obras.

O overlay AWS passa a usar `enrichment.response_retry_max_tokens: 4096`. A primeira chamada continua em 1.024 tokens; somente um corte no teto de saída permite duplicar o orçamento nos retries: **1.024 → 2.048 → 4.096 → 4.096**. JSON inválido, resposta vazia, corte abaixo do teto e limite de contexto não aumentam esse orçamento. Preflight continua em 64 tokens. O total permanece em quatro chamadas por unidade, com custos dos tokens efetivamente consumidos.

Esse teto de recuperação é operacional: não altera a identidade do dataset ou das unidades. A retomada conserva os objetos concluídos e tenta as quatro obras com erro; os novos JSONs registram a política de retry em `generation`, e o artifact guarda a configuração. Modelo, prompt, schema e teto da primeira chamada permanecem iguais. Se o novo limite também for insuficiente, a obra volta ao manifest como erro, sem aceitar JSON parcial.

Para investigar `invalid_json`, confira `validation_kind`: `json_syntax` indica sintaxe inválida; `schema_validation`, campos/tipos incompatíveis; `empty_structure`, resultado validado mas vazio. Esse último caso inclui `{"resumo":"","eventos":[]}` e campos desconhecidos ignorados pelo schema. Uma resposta curta com `stop` não identifica, sozinha, qual dessas situações ocorreu.

Após esgotar os retries, `enrichment_response_error` registra obra, fase (`unit`, `reduce` ou `final`), unidade/intervalo quando disponíveis e `response_error_bucket`/`response_error_key`. O objeto em `checkpoints/.../diagnostics/...` contém a última resposta rejeitada (até 65.536 caracteres), schema e diagnóstico. O manifest conserva esses dados em `erro_detalhes_json`; o corpo não vai para o CloudWatch, nem são capturados prompts ou credenciais. Falha ao gravar o diagnóstico mantém o erro original e registra `response_capture_error`.

Esse registro vale para novas tentativas; as respostas antigas descartadas pelo projeto não são recuperadas retroativamente. A instrumentação preserva prompts, schemas, identidade do dataset, checkpoints, limites e critérios de aceitação. Para adotá-la, aguarde o Glue encerrar e siga a mesma migração de revisão descrita acima, usando o SHA atual do checkpoint.

Em `O almada`, o diagnóstico confirmou duas unidades com apenas `I` e linhas de pontos: `s004_p01` (9.933–10.002) e `s017_p01` (18.333–18.411). A triagem do enrichment passa a ignorar unidades inteiras formadas por um marcador romano maiúsculo e uma ou mais linhas com pelo menos três pontos, sem outro conteúdo. A numeração é feita antes da triagem, preservando IDs, posições, prompts e caches das unidades com texto.

O resultado da obra registra `plan.detected_units`, `plan.planned_units`, `plan.skipped_units_count` e `plan.skipped_units` (ID, posição, motivo e hash). O log `[skip] Enrichment` resume as unidades ignoradas, que não seguem para inferência, consolidação ou índice auxiliar. O texto original permanece disponível nas camadas existentes. Uma obra sem unidades processáveis continua falhando; respostas vazias de unidades com texto também continuam rejeitadas. A retomada usa o mesmo procedimento de atualização de revisão e conserva as obras concluídas.

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
| Infraestrutura | Publicar mudanças nos `.tf`; para mudar os valores privados, atualizar o JSON e o Secret, depois executar **AWS infrastructure → plan/apply**. |
| Reprocessamento de dados | AWS ETL `run-from-raw`, mesmo Raw e novo `run_id` para outra revisão/configuração. |
| Rotação Qdrant | Atualizar o arquivo privado, executar `bootstrap_aws.py configure` e publicar novamente a API. |

Terraform, deploy e ETL compartilham a concorrência `aws-production`. Os workflows consultam o contexto atualizado no S3, sem depender de IDs EC2 antigos. O benchmark Qwen/Scout permanece separado do deploy e reutiliza a base, com DEV e holdout separados.

## Chat: leitura, escopo e primeira visita

As respostas usam Markdown no histórico e nas novas mensagens: negrito, listas, títulos, citações e tabelas. A conversão ocorre na API, com HTML e imagens desativados; mensagens do visitante continuam como texto. O renderer já fazia parte do lock de dependências e agora é uma dependência direta.

No LangGraph, `input_guard` classifica o pedido como literatura, fora de assunto ou tentativa de alterar regras/explorar o sistema. Uma aresta condicional segue para `retrieve → build_context → generate` ou para `refuse → END`, com resposta fixa e sem recuperação. O serviço HTTP mantém propriedade da conversa, limites e persistência; a decisão de caminho pertence ao grafo.

A triagem usa o mesmo provider e as mesmas vagas de concorrência, até 128 tokens de saída e as duas perguntas anteriores do checkpoint, limitadas para entender continuações. Acrescenta uma inferência curta, latência e tokens; falha de classificação bloqueia a geração com erro temporário. Cada pergunta limpa contexto, evidências e metadados transitórios, preservando as mensagens da conversa. Recusas também entram no histórico e consomem a cota de tentativa. Configuração: `chat.input_guard_enabled` e `chat.guard_max_tokens`.

Essa proteção reduz pedidos indevidos e prompt injection, sem garantir detecção de todos os ataques. O modelo não recebe ferramentas de administração. A regra vale nas chamadas a `MachadoRAG`, inclusive API e CLI; ETL e benchmarks que chamam os módulos diretamente continuam separados.

O guia abre antes da primeira consulta, explica perguntas, personas, evidências e continuidade. O aceite fica no Redis por até 30 dias, associado ao cookie da sessão; limpar o cookie ou trocar de navegador abre uma nova sessão. `Ajuda` reabre o guia. O aviso informa que consultas são registradas e recomenda evitar dados pessoais.

## Parada de emergência

No GitHub: **Actions → AWS emergency → Run workflow → main**.

| Operação | Efeito |
|---|---|
| `stop` | Grava uma trava no bucket Ops, solicita a parada dos jobs Glue ativos do projeto e da EC2; acompanha a conclusão. |
| `resume` | Inicia a EC2 e libera a trava depois de confirmar o estado `running`. Aguarde o aquecimento e teste a URL. O ETL exige novo disparo. |

O workflow usa a role de infraestrutura por OIDC, sem nova chave ou Secret. Tem concorrência própria para atender mesmo durante um ETL. A trava bloqueia novos applies, deploys e inícios de ETL pelos workflows atualizados; plano Terraform e inspeção permanecem disponíveis. Em falha parcial, a trava permanece e o workflow acusa erro: confira os serviços e repita `stop`.

EC2, volumes, snapshots, buckets, Redis e checkpoints não são excluídos. Armazenamento e IP público ainda podem ter custo; isso não é um teto de gastos AWS. Uma chamada Bedrock em andamento pode terminar antes da parada. Um Terraform/SSM já em andamento pode cruzar a última verificação; confira essas execuções e os recursos finais. Comandos administrativos diretos e workflows antigos não são bloqueados pela trava. Antes de retomar um ETL interrompido, confirme que não há Glue ativo e confira seu lock; não o apague automaticamente.

Na primeira publicação deste patch, o job de infraestrutura atualiza as permissões de leitura da trava para as roles de deploy/ETL antes da imagem. Não altere AMI, tfvars privados ou Secret para esta atualização; revise o plano normal da pipeline.

## Inspeção das consultas e do tráfego

O Redis aceita a API pela rede Docker `redis_private` e a administração pelo loopback da EC2 (`127.0.0.1:6379`). Só API e Redis participam dessa rede; Caddy e outros serviços não recebem acesso direto. O security group mantém somente HTTPS de entrada: não se abre 6379 no IP público nem para a VPC. O notebook chega ao loopback por túnel SSM autorizado por IAM, usando o perfil `terraform`; o IP do notebook não entra em uma lista pública de Redis.

Autenticar na AWS autoriza a sessão SSM conforme a política IAM; a conexão Redis ainda exige a senha da ACL `inspector`. Uma requisição web não recebe essas credenciais nem um endpoint de administração. A própria API precisa usar o Redis: o isolamento de rede não revoga esse acesso legítimo de seu processo. Não se concede acesso por estar apenas na rede interna.

Na raiz do projeto, com AWS CLI e plugin Session Manager instalados, abra o túnel no notebook:

```bash
export AWS_PROFILE=terraform
export AWS_REGION=us-east-1
python scripts/cloud.py redis-tunnel
```

Mantenha esse terminal aberto. Em outro terminal, com o mesmo perfil e ambiente Python, obtenha a senha de leitura:

```bash
python scripts/cloud.py redis-password
```

No [RedisInsight instalado no notebook](https://redis.io/docs/latest/develop/tools/insight/), adicione `127.0.0.1:16379`, usuário `inspector`, senha retornada e TLS desativado nessa conexão local. O transporte até a EC2 passa pelo túnel SSM; a porta Redis continua privada. A ACL permite leitura, inclusive `INFO`, sem operações de alteração ou `MONITOR`. O RedisInsight não está hospedado na EC2. Se usar uma instalação Docker, o `localhost` é o do container; prefira o aplicativo local para este túnel.

| Chave/filtro no Browser do RedisInsight | Conteúdo |
|---|---|
| `acartomante:session:*:chat:*:messages` | Streams com perguntas, respostas, evidências e `metadata` de geração/triagem. |
| `acartomante:usage:AAAAMMDD` | Contadores de requisições, páginas, status HTTP, respostas, recusas, tokens e custo estimado. |
| `acartomante:usage:AAAAMMDD:sessions` | HyperLogLog de sessões aproximadas; não equivale a pessoas nem a usuários simultâneos. |

Com o túnel aberto, o resumo pode ser consultado sem interface:

```bash
python scripts/dependencies.py install ops
python scripts/cloud.py usage
python scripts/cloud.py usage --day 20261017
```

Os contadores começam na publicação, usam dia UTC e expiram em 14 dias. Requisições excluem assets e health checks. Tokens/custo somam triagem e resposta dos atendimentos concluídos; chamadas falhas e retries podem ficar fora, portanto não substituem a fatura. Limpar cookies aumenta o número de sessões. As perguntas continuam sujeitas à retenção já existente do histórico, independente dos contadores.

A API registra método, rota normalizada, status e duração no log Docker, sem corpo, IP ou query string. No Console **EC2 → Instance → Monitoring**, acompanhe CPU, `NetworkIn`, `NetworkOut` e status checks. Memória/disco dentro do SO exigem instrumentação adicional. Este patch usa Redis e logs locais; não instala agente nem cria envio de logs da API para o CloudWatch. Os limites existentes continuam em seis tentativas por sessão/minuto, 300/dia globais e duas inferências simultâneas; não substituem controle financeiro.

## Dependências em um manifesto

`pyproject.toml` reúne os perfis PEP 735 `runtime`, `glue`, `ci`, `crawler`, `ops` e `tools`. Os antigos arquivos `requirements*.txt`, `.in` e `.lock` foram removidos. Os pins de produção ficam no próprio manifesto; markers de versão mantêm NumPy, SciPy e NetworkX compatíveis com os locks anteriores de Python 3.11/3.12. Ferramentas de testes/análise conservam suas faixas anteriores. O índice PyTorch continua CPU.

Use `python scripts/dependencies.py install PERFIL` dentro do venv. O leitor aplica os pins centrais também aos perfis leves e não exige instalar um novo gerenciador. `check` valida o manifesto sem instalar; `export glue --output /tmp/glue-requirements.txt` gera a entrada temporária do pip para as wheels Glue. Dockerfiles, GitHub, preparo de release no Gitea e guia de coleta consomem a mesma fonte. Não há cópia manual de listas em outros arquivos.
