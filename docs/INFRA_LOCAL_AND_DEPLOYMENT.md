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

Usei uma EC2 CPU com Redis e proxy para manter a operação pequena. A contrapartida é uma única máquina para o atendimento: atualização ou substituição pode causar indisponibilidade. Colibri e CrossEncoder continuam em CPU. `m7i-flex.large` (2 vCPUs e 8 GiB de RAM, x86_64) é o ponto inicial configurável, confirmado como elegível ao Free Tier nesta conta em `us-east-1`; o consumo reduz os créditos disponíveis. O dimensionamento ainda depende da medição de memória, latência e concorrência.

Separei IAM de runtime e ETL. O runtime lê os artefatos e usa uma chave Qdrant de leitura; o Glue escreve os dados e usa outra chave. A EC2 e seus containers compartilham a role do host, que também permite pull do ECR, operação SSM e gravação no prefixo de backups. Isso não equivale ao isolamento de roles por container oferecido por ECS.

## Repositório público e CI/CD

O guia de [bootstrap e automação AWS](AWS_AUTOMATION.md) define o que configurar, onde e quando. O Gitea continua no homelab e espelha o GitHub. O GitHub executa testes, Terraform e publicação em runners hospedados, sem acesso à rede doméstica.

Push na develop e PRs executam scanner de segredos, testes, validação dos Composes e Terraform e build/import da API. Na main, após as validações, o workflow chama o Terraform com uma role própria e aguarda o apply antes do deploy da API. A publicação da aplicação é habilitada pela variável `AWS_APP_DEPLOY_ENABLED`, inicialmente false. O ETL tem outro workflow e outra role; não roda em cada feature.

As roles OIDC exigem o ID imutável do repositório e `refs/heads/main`. Não há access keys AWS no GitHub. No modo espelhado, o bootstrap preserva proteções GitHub existentes e não acrescenta a exigência de PR que impediria o push do espelho. A revisão da main fica no fluxo do Gitea. Para um GitHub que seja a origem principal, a opção `--protect-main` aplica a proteção de PR e check validate.

## Primeira preparação, por código

Uso o perfil temporário do usuário terraform. A configuração pública fica em `infra/aws/production.tfvars.json` e é versionada. O arquivo privado `config/aws-bootstrap.env` contém o endpoint e as chaves Qdrant de produção e fica fora do Git. A EC2 e o Glue usam suas próprias roles AWS.

O bootstrap tem operações separadas para gerar a política IAM inicial, produzir o plano, aplicar o plano salvo e repetir apenas o cadastro dos segredos/variáveis quando necessário:

```bash
export AWS_PROFILE=terraform
python scripts/bootstrap_aws.py policy > infra/aws/server-terraform-policy.local.json
# Atualizar ServerTerraformIAM no IAM com o JSON gerado, conforme o guia.
python scripts/bootstrap_aws.py plan --repo SEU_USUARIO/ACartomanteRAG
python scripts/bootstrap_aws.py apply --repo SEU_USUARIO/ACartomanteRAG
```

Antes dos comandos, preencher os arquivos e preparar as ferramentas conforme [AWS_AUTOMATION.md](AWS_AUTOMATION.md). O plano é revisável; apply executa o plano salvo, sem gerar outro. Conta, repositório e configuração devem continuar correspondendo ao plano. O bucket de state fica privado, criptografado, versionado e com lock S3.

O Terraform cria VPC, subnet, regras, EC2, EBS separado, Elastic IP, buckets, ECR, roles, documentos SSM, job Glue, zona DNS e alerta mensal. Os segredos são importados para SSM SecureString fora dos argumentos Terraform. Senhas Redis já existentes são preservadas. O apply não inicia o ETL.

O cluster Qdrant Cloud existente não é recriado pelo Terraform AWS. Seu endpoint e suas duas chaves são fornecidos no bootstrap; os dados e collections de produção são gerados pelo ETL na nuvem.

## DNS sem deslocar os outros registros do domínio

Escolhi criar uma zona Route 53 apenas para o subdomínio da aplicação. `terraform output nameservers` fornece os servidores NS dessa zona. No DNS que atende `gustavodreis.dev.br`, delego **somente `acartomanterag`** para esses servidores. Se uso o DNS do Registro.br, isso é feito com registros NS do subdomínio; não substituo os servidores do domínio inteiro.

Essa delegação inicial ocorre no provedor do domínio e não é executada pelo Terraform AWS. Depois dela, o registro A e o Elastic IP ficam no código. Se o domínio já estiver em uma zona Route 53, informo seu `hosted_zone_id` e uso essa zona, sem uma nova delegação. Confiro resolução pública e eventuais registros CAA antes de emitir o certificado.

## O ETL que realmente preciso publicar

O crawler é uma coleta única no notebook, gravando PDFs e manifest diretamente no S3 Raw. É o contorno do HTTP 403 observado no acesso ao MEC pelo Glue; a mesma consulta retornou 200 no notebook. Não há migração de Bronze, Silver ou índices locais. Todas as etapas após Raw continuam no Glue. O procedimento e os limites estão em [AWS_RAW_CRAWL.md](AWS_RAW_CRAWL.md). A seleção continua fixed/256/o32 + Colibri, com recuperação Dense + CrossEncoder e collections no Qdrant Cloud.

Em GitHub Actions, o workflow AWS ETL oferece smoke, run-from-raw e o modo legado run. Ele publica um bundle da revisão escolhida e inicia o Glue com script, fontes e wheels daquela revisão. Um manifest final identifica a publicação completa; o controlador recusa bundles incompletos ou alterados e não sobrescreve artefatos existentes. A seleção por execução não exige reaplicar Terraform para trocar o código do ETL.

O smoke importa as bibliotecas, verifica S3/Qdrant, consulta o catálogo Bedrock e calcula um embedding Colibri. Não gera resposta LLM nem prova permissão de inferência. A execução do Glue tem custo. Para este corpus, run-from-raw verifica o snapshot pronto e executa bronze, silver, text-index, enrichment e enrichment-index. O crawler não é chamado. O pipeline usa o driver Python do Glue, sem distribuir essas funções em Spark.

Uso raw_id, run_id e SHA estáveis para retomar o processamento ou promover o candidato. O Raw concluído pode ser reprocessado com outro run_id sem uma nova coleta no MEC. Na promoção, o pipeline reutiliza as etapas concluídas e valida os artefatos antes de selecionar production/current.json. O documento SSM de recarga reinicia somente a API existente; antes do primeiro deploy, registra que a publicação da aplicação ainda está pendente.

O enrichment usa somente Qwen3 32B. Cada resumo concluído é persistido por prompt, configuração e texto. Deploys, reinícios, troca para Scout e benchmarks reutilizam os artefatos compatíveis. Uma interrupção entre a resposta e o checkpoint ainda pode repetir uma chamada. O ETL é acionado explicitamente; uma feature da aplicação não reenriquece o corpus.

As etapas usam lock global e checkpoint por run_id no S3. Crawler incompleto, Bronze com erro ou índices divergentes impedem promoção. Confirmar o estado do Glue antes de remover um lock de execução interrompida. A API lê a seleção publicada no startup; a recarga após promoção evita manter o catálogo anterior em memória.

## Ativando o deploy e operando depois

Depois do smoke, dos dados promovidos e do DNS delegado, habilito AWS_APP_DEPLOY_ENABLED=true nas variáveis do GitHub e executo CI and AWS deploy na main. O bootstrap já cadastrou OIDC e as demais variáveis. A configuração detalhada está em [AWS_AUTOMATION.md](AWS_AUTOMATION.md).

Nas próximas publicações, sigo as branches habituais. O workflow valida, aplica a infraestrutura por uma role própria e publica a imagem por digest via SSM. Ele lê o contexto de deploy atualizado pelo Terraform no S3, sem depender de IDs EC2 antigos em variáveis GitHub. O enrichment continua no workflow AWS ETL separado; a coleta Raw é explícita no notebook e não se repete a cada feature.

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
