# Como organizei a execução local, a AWS e a publicação

Depois de fechar o fluxo do RAG, faltava organizar a forma de deixá-lo disponível. A intenção para esta etapa era ter um endereço na minha rede local que atendesse a aplicação inteira, com os bancos e o armazenamento acessíveis pelos componentes que precisam deles.

Para isso, coloquei o Uvicorn no serviço `api` do Docker Compose. Assim, a inicialização do servidor passa a fazer parte do ambiente do projeto, junto das dependências e dos volumes já usados na instalação. O Ollama continua no notebook, preservando a instalação, os modelos e o uso da GPU.

Este documento registra esse desenho e os passos para executá-lo. A relação entre os módulos está na [arquitetura](RUNTIME_ARCHITECTURE.md); os motivos da configuração de modelos e recuperação estão nos [resultados](RESULTS_AND_LEARNINGS.md).

## Uma entrada para a aplicação

A publicação escolhida no Compose é `${API_BIND_IP}:${API_PORT}:8000`. O IP precisa ser um endereço específico da LAN. A aplicação rejeita `API_BIND_IP` com `0.0.0.0` ou `::`, para evitar a configuração de escuta em todas as interfaces.

Vale separar os dois lados desse endereço. Dentro do container, o Uvicorn escuta em `0.0.0.0:8000`, permitindo receber o tráfego encaminhado pelo Docker. No notebook, o Compose publica essa porta somente no IP informado. É esse segundo bind que escolhe o endereço exposto pelo host.

Com isso, home, chat, leitura e API ficam no mesmo endpoint. Não preciso publicar uma porta para cada tela, nem as portas administrativas das dependências.

| Serviço | Redes e acesso necessários | Porta publicada no notebook |
|---|---|---|
| `api` | Redis, Qdrant, RustFS e saída para o Ollama e downloads de modelos. | Uma, no IP LAN configurado. |
| `redis` | Rede interna `redis_private`, acessada pela API e pelos jobs. | Nenhuma. |
| `qdrant` | Rede interna `qdrant_private`, acessada pela API e pelos jobs. | Nenhuma. |
| `rustfs` | Rede interna `rustfs_private`, acessada pela API e pelos jobs; console desligado. | Nenhuma. |
| `volume-permission-helper` | Somente o volume RustFS, sem rede; encerra após o ajuste de permissões. | Nenhuma. |
| `tools` | Dependências necessárias aos jobs, ativado explicitamente pelo profile. | Nenhuma. |

Separei os backends em redes internas distintas. Redis, Qdrant e RustFS não compartilham uma rede entre si nem recebem uma rota externa por esse desenho. A API participa das três redes porque precisa conversar com cada um; os jobs usam essas conexões quando são executados.

A rede `api_egress` permite alcançar o Ollama no host e baixar modelos do Hugging Face. Ela não restringe a saída a esses destinos. Também não tratei o bind LAN como uma regra de autenticação ou de origem dos clientes: ele escolhe onde a porta é publicada, enquanto o roteamento e o firewall do host continuam influenciando quem consegue chegar a esse endereço. O notebook e a administração do Docker permanecem na fronteira de confiança desta instalação.

## Os cuidados dentro do container

A API roda com UID/GID 10001, sem capabilities, com `no-new-privileges` e filesystem somente leitura. O YAML é montado como `ro`; o cache dos modelos tem um volume próprio e `/tmp` é temporário. Esse arranjo permite as escritas necessárias aos modelos sem deixar a aplicação inteira gravável. O Docker socket não é montado.

Mantive um worker do Uvicorn. Os modelos são carregados por processo e o limite de duas gerações simultâneas também é por processo. Aumentar workers, portanto, mudaria tanto o consumo de memória quanto a concorrência efetiva; não seria apenas uma mudança na quantidade de conexões HTTP.

Os serviços persistentes têm rotação de logs, e o Uvicorn não registra cada acesso. O build exclui `.env`, volumes, notebooks e caches. Na API, desativei Swagger/OpenAPI público, listei os hosts aceitos e acrescentei os headers `nosniff`, bloqueio de framing e referrer de mesma origem. São medidas da instalação local. A entrada pública descrita adiante usa HTTPS; a sessão continua anônima, sem uma conta de usuário.

## Preparando o ambiente que já existe

O ponto de partida é o `config/.env` da instalação. Preciso manter os caminhos dos volumes de Redis, Qdrant e RustFS e suas credenciais; recriar o ambiente com pastas vazias não traria de volta os artefatos dos experimentos.

Uso `config/.env.example` como referência e confiro estes campos:

```dotenv
API_BIND_IP=<IP_LAN_REAL_DO_NOTEBOOK>
API_PORT=8000
REDIS_PASSWORD=<SEGREDO_HEXADECIMAL>
SESSION_COOKIE_SECURE=false
```

Para identificar o endereço da interface LAN:

```bash
ip -4 -brief address show scope global
```

Escolho o IP dessa interface e mantenho uma reserva no DHCP para que o endpoint não mude a cada concessão. Para gerar uma senha Redis em formato que não exija escaping adicional na URL:

```bash
python -c 'import secrets; print(secrets.token_hex(32))'
```

O Compose fornece os nomes internos `redis`, `qdrant` e `rustfs` à API e aos jobs. Dentro de um container, `localhost` representa aquele próprio container, então não uso o loopback do notebook como endereço desses serviços. `API_EXTRA_HOSTS` permite acrescentar um hostname LAN aos hosts aceitos pela aplicação; o wildcard total `*` é rejeitado.

## Fazendo o container chegar ao Ollama

Aqui havia um detalhe de rede que precisava de atenção. Se o Ollama escutasse somente em `127.0.0.1`, a API em Docker não conseguiria alcançá-lo por `host.docker.internal`. Deixá-lo em todas as interfaces também o disponibilizaria na LAN, além da entrada que escolhi para a aplicação.

Para esta instalação Linux, a configuração prevista é o bind do Ollama no gateway da bridge padrão do Docker. Primeiro consulto o endereço e abro o override do serviço:

```bash
docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'
sudo systemctl edit ollama.service
```

No editor, uso o IP que o primeiro comando retornou:

```ini
[Service]
Environment="OLLAMA_HOST=<GATEWAY_DOCKER>:11434"
```

Depois aplico a configuração e confiro os modelos nesse mesmo endereço:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
OLLAMA_HOST=<GATEWAY_DOCKER>:11434 ollama list
```

O modelo esperado é `qwen3.5:4b`. Se precisar baixá-lo, uso o mesmo `OLLAMA_HOST` no comando `ollama pull qwen3.5:4b`. Clientes do próprio notebook que usavam `localhost:11434` também precisam passar a usar o endereço escolhido.

No Compose, `host.docker.internal` é associado a `host-gateway`, e `OLLAMA_BASE_URL_DOCKER` assume `http://host.docker.internal:11434` por padrão. A conectividade entre esse mapeamento e o bind escolhido precisa ser conferida na instalação real. Os comandos acima são passos de implantação; aplicar um patch de documentação não os executa.

## Subindo e distinguindo processo ativo de dependências prontas

Na raiz do projeto, valido a configuração antes de construir e iniciar os serviços:

```bash
docker compose --env-file config/.env config --quiet
docker compose --env-file config/.env up -d --build
docker compose --env-file config/.env ps
docker compose --env-file config/.env logs --tail 60 api
```

O endereço de entrada é `http://<IP_LAN_REAL_DO_NOTEBOOK>:8000`, considerando a porta padrão. O primeiro carregamento pode demorar porque envolve os modelos; o acesso ao Hugging Face e espaço no cache precisam estar disponíveis. Os volumes de dados continuam necessários, pois os fontes do projeto não incluem os manifests e collections da instalação.

Pelo mesmo endpoint, faço duas verificações com propósitos diferentes:

```bash
curl -fsS http://<IP_LAN_REAL_DO_NOTEBOOK>:8000/health
curl -fsS http://<IP_LAN_REAL_DO_NOTEBOOK>:8000/ready
```

`/health` é a checagem leve de liveness usada pelo Docker. `/ready` verifica Redis, existência do bucket de índices, Ollama e collection textual. A checagem do Ollama consulta o inventário de modelos. No Bedrock, consulto os metadados do modelo ou do perfil de inferência. Nenhuma dessas verificações gera tokens; elas também não demonstram a qualidade da resposta nem provam permissão para uma chamada de inferência. Um processo responder ao healthcheck não é, por si só, uma comprovação de que todas as dependências estão disponíveis.

Nos logs, a configuração esperada aparece como `runtime_ready strategy=dense_cross_encoder index=idx_colibri-embed-ptbr_2b279b323307`. Essa linha ajuda a conferir qual índice a API realmente carregou, além do que está escrito no YAML.

## Como continuo executando os jobs

ETL, geração de resumos e benchmarks continuam separados do startup da API. Para executá-los, uso o serviço do profile `tools`, que contém as dependências de análise e acessa as mesmas redes internas, sem abrir outro servidor.

```bash
docker compose --env-file config/.env run --rm tools python -m src.generation.cli 'Quem é Capitu?'
docker compose --env-file config/.env run --rm tools python -m src.enrichment.work_summary --title 'O Alienista' --model qwen3.5:4b --upload
```

Para outras tarefas, troco o módulo e consulto o respectivo `--help`. A ordem do pipeline principal permanece Raw → Bronze → Silver → índice. O enrichment tem sua própria construção de índices em `src/indexing/build_enrichment_embedding_indexes.py`.

Construir um índice produz um candidato. Antes de torná-lo o índice atendido pela aplicação, confiro manifest e collection, publico explicitamente e ajusto a seleção por `index_id` ou release. Com o pin atual, mudar somente a release não troca o índice; esse comportamento está detalhado na [arquitetura](RUNTIME_ARCHITECTURE.md).

Os benchmarks chamam os módulos diretamente e usam configurações da execução, sem passar pela API. Salvo os resultados, configurações e manifests no bucket de benchmarks para comparar depois. Um arquivo criado apenas dentro de um container `--rm` desaparece com ele; os uploads para RustFS precisam ser explícitos.

## Versões e configuração da publicação

A base da imagem é `python:3.12-slim`. No Compose estão `redis/redis-stack-server:7.4.0-v8`, `qdrant/qdrant:v1.19.1` e `rustfs/rustfs:1.0.0`. As entradas em `requirements-runtime.txt` mantêm as faixas aceitas, mas o build agora consome `requirements-runtime.lock`, com versões resolvidas para Python 3.12/CPU. O Glue tem um lock separado para Python 3.11. As revisões de Colibri, tokenizer e reranker na configuração cloud foram fixadas por SHA. Imagens base e auxiliares ainda usam tags de versão; por isso não descrevo os builds como idênticos bit a bit. A imagem publicada é selecionada por digest no ECR.

Mantenho a entrada LAN para o trabalho local. Na AWS, uso o desenho abaixo, com configuração própria e cookie seguro para HTTPS.

## O que muda quando levo o projeto para a AWS

Quis manter os mesmos módulos de preparação e recuperação, mudando os serviços que preciso operar. Por isso, o nome `RustFSManager` continua no código por compatibilidade, mas sua implementação é uma interface S3: na AWS não subo um servidor RustFS. O `boto3` usa a role IAM da EC2 ou do Glue, sem colocar access keys nas imagens ou no GitHub.

| Responsabilidade | Local | AWS |
|---|---|---|
| Serviços | `docker-compose.local.yml`, que inclui o Compose LAN existente | `docker-compose.aws.yml` |
| Armazenamento | RustFS nas redes internas | S3 privado e versionado |
| Vetores | Qdrant local | Cluster Qdrant Cloud existente |
| Geração | Ollama, Qwen3.5 4B | Bedrock, Qwen3 32B como padrão; Scout configurável |
| Entrada | IP LAN específico | Caddy, HTTPS 443 no IP privado específico associado ao Elastic IP |
| Conversas | Redis Stack interno | Redis Stack em EBS separado; inspeção no loopback por túnel SSM |
| Jobs | Profile `tools` | Glue 5.1, execução explícita no driver Python |

O domínio escolhido é `https://acartomanterag.gustavodreis.dev.br`, em `us-east-1`. O security group não abre SSH, HTTP, Redis ou uma porta própria da API. Caddy usa o desafio TLS-ALPN-01 na 443, e a API permanece sem porta publicada. `/health` e `/ready` ficam disponíveis internamente, mas o proxy não os expõe.

Usei uma EC2 CPU com Redis e proxy para manter a operação pequena. A contrapartida é uma única máquina para o atendimento: atualização ou substituição pode causar indisponibilidade. Colibri e CrossEncoder continuam em CPU. `t3.large` é um ponto inicial configurável, ainda sujeito à medição de memória, latência e concorrência; não é um dimensionamento validado.

Separei IAM de runtime e ETL. O runtime lê os artefatos e usa uma chave Qdrant de leitura; o Glue escreve os dados e usa outra chave. A EC2 e seus containers compartilham a role do host, que também permite pull do ECR, operação SSM e gravação no prefixo de backups. Isso não equivale ao isolamento de roles por container oferecido por ECS.

## Repositório público e CI/CD

Posso publicar as fontes como portfólio. A proteção vem das permissões e credenciais, e o histórico do Git também precisa passar pela auditoria. `.env`, estados e planos Terraform, credenciais, volumes e dados de sessão ficam fora do Git. Se uma credencial real já tiver sido publicada, ignorar o arquivo depois não a revoga.

O GitHub Actions é quem executa o CI/CD da AWS. O Gitea do homelab não participa desse deploy. O workflow usa runners hospedados pelo GitHub; PRs públicos não ganham uma conexão com a minha rede doméstica.

- Push na `develop` e PRs para `develop`/`main`: scanner de segredos, testes, sintaxe, configuração dos Composes, Terraform e build/import da API.
- Push ou execução manual na `main`: as mesmas validações; depois, imagem no ECR e bundles no S3; por fim, deploy via documento SSM restrito.
- A proteção da `main` exige PR, check `validate` atualizado, histórico linear e conversas resolvidas; bloqueia force push e exclusão, inclusive para administradores. Não exijo uma aprovação de outra pessoa nesta configuração de mantenedor único.
- A role OIDC exige o ID imutável do repositório e `refs/heads/main`. Não guardo chaves AWS permanentes no GitHub. O job de validação não recebe essa role.

A proteção de branch é uma configuração do GitHub, não um efeito de adicionar um YAML. Eu a aplico com `scripts/configure_github.py`, usando `gh` autenticado com permissão administrativa. O script também cria `develop` caso ela não exista e configura as variáveis públicas de deploy a partir dos outputs Terraform. Não altera a visibilidade do repositório.

## Primeira preparação, por código

No notebook preciso de AWS CLI com SSO/profile autorizado, Terraform 1.11+, GitHub CLI, Python e Docker/Compose. Os comandos abaixo são de implantação: aplicar o ZIP não os executa. Os arquivos `.tfvars` são escolhas reais de conta e orçamento, preenchidas antes de planejar qualquer gasto.

```bash
aws sso login --profile SEU_PROFILE
export AWS_PROFILE=SEU_PROFILE
export AWS_REGION=us-east-1
python -m pip install boto3

python scripts/configure_github.py --repo SEU_USUARIO/ACartomanteRAG --inspect
cp infra/aws/terraform.tfvars.example infra/aws/terraform.tfvars
# Preencher e-mails, ID numérico do repo e revisão do bundle Glue.

python infra/bootstrap_state.py --bucket NOME_GLOBAL_UNICO_PARA_ESTADO
terraform -chdir=infra/aws init -backend-config=backend.hcl
terraform -chdir=infra/aws plan -out=deployment.tfplan
terraform -chdir=infra/aws apply deployment.tfplan
terraform -chdir=infra/aws output -json > infra/aws/outputs.local.json
python scripts/cloud.py --outputs infra/aws/outputs.local.json secrets
```

O bootstrap cria o bucket de estado com versionamento, criptografia e bloqueio de acesso público. Terraform usa lock de estado no S3. Os segredos de Qdrant e Redis são gravados como SecureString no SSM por entrada protegida no terminal; não são recursos contendo senhas no estado Terraform. Reexecutar `secrets` preserva as senhas Redis já existentes.

O Terraform cria a VPC, subnet, regras, EC2, volume EBS separado, Elastic IP, buckets, ECR, roles, documento SSM, job Glue, zona DNS e alerta mensal. Buckets e volume têm proteção contra destruição no código. Após o primeiro plano, posso fixar o `ami_id` resolvido para repetir a imagem do sistema operacional.

O cluster Qdrant não é recriado pelo Terraform AWS. Para também gerenciá-lo por código, há `scripts/import_qdrant.py`: ele usa o provider oficial para gerar a configuração do cluster real, preservando seu pacote gratuito. Exige a chave de administração Qdrant Cloud, diferente das chaves do banco. Recusa um plano com atualização, criação ou exclusão e acrescenta `prevent_destroy` ao recurso gerado.

```bash
python scripts/import_qdrant.py --account-id UUID_DA_CONTA --cluster-id UUID_DO_CLUSTER
# Conferir infra/qdrant/cluster.generated.tf e o plano; então persistir o import:
python scripts/import_qdrant.py --account-id UUID_DA_CONTA --cluster-id UUID_DO_CLUSTER --apply-import
```

Essa importação ainda precisa dos IDs e da chave reais. A configuração gerada e o lock do provider devem ser versionados; seu estado fica em outra chave do mesmo backend privado. Collections e conteúdo continuam sob responsabilidade do ETL Python.

## DNS sem deslocar os outros registros do domínio

Escolhi criar uma zona Route 53 apenas para o subdomínio da aplicação. `terraform output nameservers` fornece os servidores NS dessa zona. No DNS que atende `gustavodreis.dev.br`, delego **somente `acartomanterag`** para esses servidores. Se uso o DNS do Registro.br, isso é feito com registros NS do subdomínio; não substituo os servidores do domínio inteiro.

Essa delegação inicial ocorre no provedor do domínio e não é executada pelo Terraform AWS. Depois dela, o registro A e o Elastic IP ficam no código. Se o domínio já estiver em uma zona Route 53, informo seu `hosted_zone_id` e uso essa zona, sem uma nova delegação. Confiro resolução pública e eventuais registros CAA antes de emitir o certificado.

## O ETL que realmente preciso publicar

Não levo a matriz de experimentos para o job de produção. A seleção é `fixed/256/o32 + Colibri`; a recuperação continua Dense + CrossEncoder. Tenho dois caminhos: migrar os artefatos aprovados e reutilizar a collection textual, ou executar crawler → Raw → Bronze → Silver → índice quando precisar reconstruir os dados.

`scripts/package_release.py` produz um ZIP de fontes sem `.env` e dados. O Dockerfile de wheels usa Python 3.11 em Amazon Linux 2023, compatível com Glue 5.1, e baixa as dependências CPU fixadas. Glue continua sendo um teste de compatibilidade a concluir: o pipeline usa o driver Python, sem transformar essas funções em operações Spark distribuídas. Duas workers são o mínimo configurado para esse tipo de job, mesmo sem paralelismo Spark útil.

```bash
# Primeiro versionar as fontes desta entrega no Git, sem .env, dados ou tfvars.
revision=$(git rev-parse HEAD)
# Os SHAs dos modelos já estão fixados; lock_models.py só resolve revisões não fixadas.
python scripts/lock_models.py
docker build -f infra/glue/Dockerfile.wheels --output type=local,dest=dist/glue .
python scripts/package_release.py --sha "$revision"
python scripts/cloud.py --outputs infra/aws/outputs.local.json upload-glue --revision "$revision"
# glue_bundle_prefix no tfvars deve ser glue/<essa revisão>.
terraform -chdir=infra/aws plan -out=deployment.tfplan
terraform -chdir=infra/aws apply deployment.tfplan

python scripts/cloud.py --outputs infra/aws/outputs.local.json glue --revision "$revision" --run-id smoke-v1 --smoke --wait
```

O smoke importa as bibliotecas, testa S3 e Qdrant, consulta o catálogo Bedrock e calcula um embedding Colibri. Não chama o LLM, mas a execução do Glue tem custo. Uma chamada de inferência real e o benchmark continuam necessários antes da abertura pública.

Para aproveitar os dados locais, a migração tem plano por padrão. Leva Raw, a Bronze referenciada, a Silver escolhida e o manifest textual; não leva a matriz de embeddings ou os resumos do gerador anterior. Compara bytes existentes e recusa overwrite divergente. O exemplo abaixo roda no profile `tools`, com as credenciais temporárias do meu profile AWS acessíveis pelo mount somente leitura:

```bash
aws sso login --profile SEU_PROFILE
docker compose --env-file config/.env -f docker-compose.local.yml run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$HOME/.aws:/tmp/.aws:ro" \
  -v "$PWD/config/.env:/tmp/source.env:ro" \
  -v "$PWD/infra/aws/outputs.local.json:/tmp/outputs.json:ro" \
  -e HOME=/tmp -e AWS_PROFILE=SEU_PROFILE -e AWS_CONFIG_FILE=/tmp/.aws/config \
  -e AWS_SHARED_CREDENTIALS_FILE=/tmp/.aws/credentials \
  tools python -m src.etl.migrate_storage \
  --source-env /tmp/source.env --source-endpoint http://rustfs:9000 --outputs /tmp/outputs.json
# Repetir o comando com --apply após conferir o plano.
```

O SDK usa o cache SSO em `/tmp/.aws/sso/cache`. Faço `aws sso login` antes, para que o cache temporário montado somente leitura esteja válido. Não salvo essas credenciais no bundle nem na imagem.

Com a migração concluída e a collection textual já no Qdrant Cloud:

```bash
python scripts/cloud.py --outputs infra/aws/outputs.local.json glue --revision "$revision" \
  --run-id corpus-v1 --steps text-index,enrichment,enrichment-index \
  --reuse-index-id idx_colibri-embed-ptbr_2b279b323307 --promote --wait
```

Se for uma reconstrução completa, omito `--reuse-index-id` e uso `--steps crawl,bronze,silver,text-index,enrichment,enrichment-index`. Sem `--promote`, o resultado continua candidato.

O enrichment usa somente Qwen3 32B. O dataset é determinístico e retomável; cada resumo concluído é persistido imediatamente por prompt, configuração e texto. Deploys, reinícios, troca para Scout e benchmarks reutilizam esses artefatos. Uma interrupção entre receber a resposta e salvar o checkpoint ainda pode repetir aquela chamada; não prometo cobrança exatamente uma vez.

As etapas usam um lock global de ETL e um checkpoint por `run-id`. Crawler incompleto, Bronze com erro ou índices com contagens divergentes impedem a promoção. As collections usam IDs determinísticos nos pontos para retomar upserts sem duplicar registros. Reinício do host não limpa um lock de job interrompido: primeiro confiro o estado do Glue e só depois removo explicitamente `production/etl.lock` no bucket de índices, se não houver execução ativa.

A promoção publica um único manifest `production/current.json` com o índice textual e os ponteiros da Bronze, Silver e enrichment. Os candidatos têm caminhos próprios, então preparar uma versão não troca o catálogo em uso. Cada processo da API lê a seleção uma vez no startup. Assim evito misturar o catálogo de uma execução com o índice de outra.

## Ativando o deploy e operando depois

Depois do smoke, dos dados promovidos e do DNS delegado, configuro o GitHub:

```bash
terraform -chdir=infra/aws output -json > infra/aws/outputs.local.json
python scripts/configure_github.py --repo SEU_USUARIO/ACartomanteRAG \
  --terraform-output infra/aws/outputs.local.json
```

O primeiro commit da implantação entra pelo fluxo de branches existente; depois, o caminho normal é trabalhar na `develop`, validar o PR para `main` e fazer o merge. O workflow da `main` publica a imagem por digest e usa SSM para atualizar a máquina. Ele não roda `terraform apply`, crawler ou enrichment automaticamente. Mudanças de infraestrutura continuam usando `plan`/`apply` por CLI; o GitHub tem uma role limitada ao deploy da aplicação, sem administração IAM.

Antes de uma atualização, salvo um RDB consistente do Redis. Se a nova release falhar no readiness ou HTTPS, o aplicador de deploy tenta restabelecer a imagem anterior. O volume EBS e o Elastic IP são independentes da EC2; a substituição do host remonta o volume e reutiliza a última release. Há um timer de backup diário para S3, com retenção de 30 dias.

```bash
python scripts/cloud.py --outputs infra/aws/outputs.local.json redis-password
python scripts/cloud.py --outputs infra/aws/outputs.local.json redis-tunnel
# RedisInsight: localhost:16379, usuário inspector, senha REDIS_INSPECT_PASSWORD do SSM.
# A senha é diferente da senha administrativa; a ACL do inspector permite leitura.

python scripts/cloud.py --outputs infra/aws/outputs.local.json admin backup
python scripts/cloud.py --outputs infra/aws/outputs.local.json admin restore --key backups/ANO/MES/DIA/HORARIO.rdb
python scripts/cloud.py --outputs infra/aws/outputs.local.json data-rollback --run-id RELEASE_ANTERIOR
python scripts/cloud.py --outputs infra/aws/outputs.local.json admin restart
```

O túnel exige o plugin Session Manager instalado no notebook e uma identidade AWS autorizada. A porta 6379 continua fechada para a internet. A restauração valida o RDB, preserva a pasta Redis anterior e reconstrói o AOF antes de voltar à configuração normal.

Os registros de resposta guardam persona, modelo, índice, tokens, custo estimado e tempos no campo `metadata` dos streams Redis. As sessões são anônimas: não as conto como pessoas identificadas. Configurei também limites de seis consultas por sessão/minuto, 300 tentativas por dia para a aplicação inteira e duas gerações simultâneas, ajustáveis no YAML cloud. Falhas também consomem a cota de tentativa. Esses limites não substituem um orçamento financeiro; as tarifas no YAML são referências, não uma leitura da fatura AWS.

## O que falta medir antes de chamar isso de produção validada

Esta entrega prepara as etapas de aplicação, infraestrutura e dados. A comparação Qwen × Scout nos HTMLs, a latência real, o consumo com concorrência e os testes reais de restauração ficam para a etapa de validação da publicação. Não reapresento os resultados antigos como se fossem medições do Bedrock.

As verificações locais cobrem os contratos de configuração, a serialização Converse, limites, retomada de resumos, migração e empacotamento. A configuração dos Composes e do Caddy é validável sem subir servidores. Build Docker, execução Glue, import Qdrant e chamadas AWS dependem do ambiente real. O CI inclui as verificações de build e Terraform antes de permitir deploy.

Referências de implementação: [GitHub OIDC](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws), [claims imutáveis](https://docs.github.com/en/actions/reference/security/oidc), [Qwen no Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-qwen-qwen3-32b.html), [soft switch de reasoning do Qwen](https://huggingface.co/Qwen/Qwen3-32B), [dependências Glue](https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-python-libraries.html), [backend S3](https://developer.hashicorp.com/terraform/language/backend/s3) e [provider Qdrant](https://registry.terraform.io/providers/qdrant/qdrant-cloud/latest/docs/resources/accounts_cluster).

## Raspberry Pi: Ollama dentro do Docker

**Situação em 15/10/2026: execução e deploy locais suspensos por falta de espaço no Raspberry.** Os workflows do Gitea executam somente checagens estáticas enquanto aguardo a manutenção. O espelhamento para o GitHub e o CI/CD da AWS permanecem independentes. Consulte [impeditivos e critérios de retomada](IMPEDITIVOS.md) antes de usar os comandos de runtime desta seção.

No Raspberry uso o Compose base junto de `docker-compose.raspberry.yml`. O notebook continua usando o Ollama do host com o Compose local anterior. Na AWS a configuração permanece Bedrock.

O overlay do Raspberry aponta API e tools para `http://ollama:11434`, sem publicar a porta 11434 no host. O servidor tem rede privada para seus clientes e uma rede de saída para baixar o modelo. Os pesos ficam no volume nomeado `ollama_models` do projeto Compose. Mantenho o mesmo `COMPOSE_PROJECT_NAME` entre deploys para reutilizar esse volume e o cache da API.

O serviço `ollama-model` verifica `qwen3.5:4b` e faz pull somente quando ele ainda não existe. A API depende da conclusão com sucesso desse preparo. Falha de download mantém a API bloqueada e deixa um diagnóstico curto. O primeiro download tem aproximadamente 4 GB; espaço ocupado em disco não representa a RAM total necessária ao modelo e ao restante do RAG.

Uso a imagem oficial `ollama/ollama:0.40.4`, cujo manifest oferece Linux ARM64 e AMD64. Não adiciono devices de GPU ao container do Raspberry. Limito Ollama a um modelo carregado e uma geração simultânea; o overlay de configuração também deixa uma chamada simultânea na API e amplia o timeout para CPU. Esses limites não constituem dimensionamento validado no hardware real.

Na raiz do checkout, mantendo o `config/.env` privado e seus volumes:

```bash
# Se opero o projeto usado pelo workflow de produção:
export COMPOSE_PROJECT_NAME=acartomante-prod
bash scripts/compose_raspberry.sh config --quiet
bash scripts/compose_raspberry.sh up --detach --build --wait --wait-timeout 3600 --remove-orphans
bash scripts/compose_raspberry.sh ps
bash scripts/compose_raspberry.sh exec -T ollama ollama list
```

O wrapper usa `config/.env` quando esse arquivo existe; `LOCAL_ENV_FILE` permite selecionar outro arquivo privado. No runner sem esse arquivo, as variáveis continuam vindo do job. O bind LAN continua obrigatório em produção. Somente projetos temporários `acartomante-ci-*` recebem loopback como fallback para validar a configuração.

Para selecionar outro modelo, defino `OLLAMA_MODEL` no ambiente privado e repito o comando de atualização. O mesmo valor é usado no download, chat e enrichment. O padrão permanece `qwen3.5:4b`; trocar para um modelo menor altera o comportamento do baseline.

Durante a manutenção, os jobs Integration CI/Main PR e o job de produção mantêm seus nomes e eventos, mas validam somente a sintaxe Python/shell e as configurações Compose. Os passos de pull, build, inicialização, promoção e limpeza de serviços estão explicitamente desabilitados nos workflows. O sucesso desses checks não valida a infraestrutura nem publica uma release. Após a manutenção, a retomada precisa restaurar as verificações reais de runtime e o deploy, conforme os critérios em [IMPEDITIVOS.md](IMPEDITIVOS.md). O RAG completo exige os manifests, datasets e collections aprovados nos volumes da instalação.

Para diagnóstico:

```bash
bash scripts/compose_raspberry.sh logs --tail 80 ollama ollama-model api
```

Não uso `down --volumes` na instalação persistente. Essa limpeza fica restrita aos projetos temporários dos workflows CI.

Referências: [Docker oficial do Ollama](https://docs.ollama.com/docker), [Qwen3.5 4B](https://ollama.com/library/qwen3.5:4b), [release Ollama 0.40.4](https://github.com/ollama/ollama/releases/tag/v0.40.4) e [ordem de inicialização Compose](https://docs.docker.com/compose/how-tos/startup-order/).
