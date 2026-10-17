# ACartomanteRAG

Desenvolvimento de uma aplicação RAG nas obras de Machado de Assis.

O objetivo do projeto é montar um sistema especialista documental de ponta a ponta, da obtenção dos documentos à integração com um LLM capaz de responder perguntas sobre as obras.

Machado de Assis foi escolhido por ser um dos meus autores preferidos e por suas obras estarem disponíveis em domínio público.

Isso cria uma oportunidade para o desenvolvimento de um sistema de conhecimento que vai da obtenção dos dados via crawler até a extração inteligente de informações por meio de uma arquitetura RAG.

Dada essa oportunidade, espero, neste projeto, trabalhar:

* conceitos de LLMOps;
* Data Science aplicada a documentos;
* diferentes estratégias de recuperação de informações em documentos longos;
* roteamento de modelos para requisições;
* criação de pipelines de dados reproduzíveis;
* CI/CD para repositório Git e publicação de aplicações e features;
* observabilidade e infraestrutura utilizando soluções open source.

Dadas as minhas intenções, algumas ideias permeiam as decisões do projeto em nível de arquitetura.

> "Limitações estimulam a criatividade."

De maneira autoimposta, quero usar modelos open source de IA durante o projeto e na extração dessas informações. Esses modelos, como o Llama 3 8B, rodam localmente na minha máquina e, francamente, são ruinzinhos para grande parte das coisas.

Mas, neste projeto, quero contornar essas dificuldades e utilizá-los de forma que possam se tornar bons sistemas e agentes.

A ideia é desenvolver a arquitetura do projeto ao redor de suas limitações, compensando-as com outras soluções, como:

* estratégias de chunking;
* heurísticas textuais;
* validação.

Acredito que isso nos força a trabalhar melhor com o que temos e a construir dados de maior qualidade, que, consequentemente, melhoram as IAs utilizadas.

Parte do que vem sendo feito no pré-processamento de dados já envolve isso. Uso extensivamente heurísticas para extrair capítulos e classificar obras, sendo que um LLM maior e mais potente provavelmente faria isso melhor e mais rápido, e eu não teria dores de cabeça nem notebooks com mais de 100 células.

Mas também não teria o conhecimento adquirido ao trabalhar nesses cenários em que a descrição e a estrutura importam, inclusive em contextos com modelos proprietários. Ao escolher usar essas abordagens locais, mantenho a privacidade dos meus documentos e o domínio completo da solução.

## Por que LLMs locais?

* LLMs locais são ótimas ferramentas e podem ser usados em cenários nos quais os documentos não podem sair da organização;
* quando privacidade é um requisito;
* quando precisamos de mais controle sobre a solução;
* quando o custo se torna mais relevante.

Isso é útil para todo caso? Não. Mas, dependendo do contexto, pode ser.

A graça de um projeto desses é aprender, quebrar as coisas, consertar, subir e fazer de novo!

## Metas

* Plataforma web para chat sobre obras de Machado de Assis;
* desenvolvimento de um jogo com as informações extraídas, em uma espécie de quiz;
* comparação entre abordagens RAG:
  * diferentes arquiteturas de recuperação;
  * diferentes modelos;
* pipeline de CI/CD para publicação;
* uso de soluções open source para a parte de DevOps do projeto.

## Atalhos de desenvolvimento

Execute `make help` na raiz do projeto. O Makefile reaproveita os Composes, scripts e workflows existentes.

| Comando | Ação |
|---|---|
| `make local` | Valida, constrói e inicia a aplicação no notebook; mostra a URL publicada. |
| `make local-ready` | Consulta as dependências em `/ready` no IP e porta reais. |
| `make local-logs` / `make local-status` | Acompanha logs / lista containers. |
| `make local-restart` | Recria somente a API para aplicar variáveis; mudanças de código usam `make local`. |
| `make local-stop` / `make local-down` | Para serviços / remove containers e redes, preservando volumes. |
| `make raspi` | Executa checagens estáticas; a suspensão por falta de espaço permanece. |
| `make aws` | Dispara CI, Terraform e deploy na `main` já espelhada no GitHub. |
| `make aws-status` | Lista execuções do GitHub; `aws-watch RUN=<id>` acompanha e `aws-logs RUN=<id>` mostra logs. |
| `make aws-plan` / `make aws-apply` | Dispara plano / plano e apply da infraestrutura pelo GitHub. |
| `make aws-stop` / `make aws-resume` | Solicita emergência / retomada; consulte as limitações em [AWS_AUTOMATION.md](docs/AWS_AUTOMATION.md#parada-de-emergência). |
| `make deps` / `make test` / `make check` | Instala o perfil CI / executa testes / valida sintaxe, Composes e dependências. |

No notebook, `config/.env` continua definindo credenciais, volumes e bind; Ollama e índices locais precisam estar disponíveis. Para restringir o teste ao próprio notebook, mantenha `API_BIND_IP=127.0.0.1` nesse arquivo ou use `make local API_BIND_IP=127.0.0.1`. A inicialização dos modelos continua após o `up`; confirme com `make local-ready`.

Se o Docker exige sudo, use `make local DOCKER="sudo docker"`, repetindo essa opção nos outros atalhos Docker. Para trocar o arquivo privado, use `ENV_FILE=/caminho/ambiente.env`; para logs, `SERVICE=api LOG_LINES=100`. Dependências e testes aceitam `PYTHON=.venv/bin/python`; `make deps PROFILE=ops` seleciona outro perfil do mesmo `pyproject.toml`.

Após manutenção e revisão de capacidade, execute **no Raspberry** `make raspi-up RASPI_RUNTIME=1`; esse comando inclui a imagem e o modelo Ollama antes suspensos. `raspi-status`, `raspi-logs`, `raspi-stop` e `raspi-down` também atuam no Docker do host atual. O Makefile não estabelece SSH com o Raspberry.

Os atalhos AWS usam `gh` autenticado e o repositório `Gusreis7/ACartomanteRAG`, alterável por `GITHUB_REPO`. Primeiro publique o código pelo Gitea e confira o espelho: `make aws` usa a revisão presente na `main` do GitHub, sem enviar arquivos locais. A pipeline mantém OIDC, o Secret privado Terraform e `AWS_APP_DEPLOY_ENABLED`; não são criadas novas chaves AWS no notebook. Os atalhos não mudam os gatilhos existentes: um push na `main` já dispara CI/CD. Se o push já iniciou a revisão desejada, use `make aws-status` e acompanhe essa execução, evitando outro disparo.

## Execução e documentação

| Documento | Conteúdo |
|---|---|
| [Arquitetura e fluxo](docs/RUNTIME_ARCHITECTURE.md) | Pipeline, módulos, estado, chat, evidências e UI. |
| [Execução e publicação](docs/INFRA_LOCAL_AND_DEPLOYMENT.md) | Ambiente local, rede, AWS, ETL na nuvem e CI/CD. |
| [AWS: bootstrap e automação](docs/AWS_AUTOMATION.md) | O que configurar, primeira infraestrutura, ETL integral na nuvem e publicação pelo GitHub. |
| [Resultados e decisões](docs/RESULTS_AND_LEARNINGS.md) | Benchmarks, configuração promovida e limites dos resultados. |

No ambiente local, mantenho o `config/.env` existente, usando `config/.env.example` como referência. Preservo os volumes e as credenciais e defino `API_BIND_IP` com o IP LAN do notebook.

```bash
docker compose --env-file config/.env -f docker-compose.local.yml config --quiet
docker compose --env-file config/.env -f docker-compose.local.yml up -d --build
```

A aplicação fica em `http://<IP_LAN>:8000`. Na AWS, uso `docker-compose.aws.yml`, Terraform, S3, Qdrant Cloud e Bedrock. Desenvolvimento e PRs ficam no Gitea; o GitHub espelhado valida e publica somente a `main` ou por disparo manual.

As dependências ficam em um único `pyproject.toml`, organizadas por perfil. Com Python 3.11+ e o ambiente virtual ativo:

```bash
python scripts/dependencies.py install runtime  # API/RAG, CPU
python scripts/dependencies.py install ci       # testes, sem instalar os modelos
```

Os perfis `glue`, `crawler`, `ops` e `tools` atendem às outras operações. O mesmo leitor é usado pelos Dockerfiles e workflows; os sete arquivos antigos de requirements foram removidos. Os pins de API/Glue foram preservados, inclusive as diferenças entre Python 3.11 e 3.12. Não é necessário instalar todos os perfis no notebook.

A preparação da infraestrutura, o ETL integral na nuvem, o reaproveitamento do enrichment e o CI/CD estão no guia de [bootstrap e automação AWS](docs/AWS_AUTOMATION.md). Aplicar o patch não cria recursos nem executa deploy.
