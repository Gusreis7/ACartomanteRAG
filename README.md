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

## Execução e documentação

| Documento | Conteúdo |
|---|---|
| [Arquitetura e fluxo](docs/RUNTIME_ARCHITECTURE.md) | Pipeline, módulos, estado, chat, evidências e UI. |
| [Execução e publicação](docs/INFRA_LOCAL_AND_DEPLOYMENT.md) | Ambiente local, rede, AWS, migração de dados e CI/CD. |
| [Resultados e decisões](docs/RESULTS_AND_LEARNINGS.md) | Benchmarks, configuração promovida e limites dos resultados. |

No ambiente local, mantenho o `config/.env` existente, usando `config/.env.example` como referência. Preservo os volumes e as credenciais e defino `API_BIND_IP` com o IP LAN do notebook.

```bash
docker compose --env-file config/.env -f docker-compose.local.yml config --quiet
docker compose --env-file config/.env -f docker-compose.local.yml up -d --build
```

A aplicação fica em `http://<IP_LAN>:8000`. Na AWS, uso `docker-compose.aws.yml`, Terraform, S3, Qdrant Cloud e Bedrock. O GitHub Actions executa as validações na `develop` e o deploy após o merge na `main`. O Gitea permanece no homelab, sem participar do deploy AWS.

A preparação da infraestrutura, a migração dos artefatos, a geração única do enrichment e a proteção da branch estão no guia de [execução e publicação](docs/INFRA_LOCAL_AND_DEPLOYMENT.md). Aplicar o patch não cria recursos nem executa deploy.
