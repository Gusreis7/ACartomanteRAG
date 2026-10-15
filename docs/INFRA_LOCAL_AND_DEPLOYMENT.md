# Como organizei a execução local e a rede

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

Os serviços persistentes têm rotação de logs, e o Uvicorn não registra cada acesso. O build exclui `.env`, volumes, notebooks e caches. Na API, desativei Swagger/OpenAPI público, listei os hosts aceitos e acrescentei os headers `nosniff`, bloqueio de framing e referrer de mesma origem. São medidas que acompanham a execução local escolhida, sem substituir a etapa futura de publicação com HTTPS e autenticação.

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

`/health` é a checagem leve de liveness usada pelo Docker. `/ready` verifica Redis, existência do bucket de índices, Ollama e collection textual. Na checagem do Ollama há uma pequena geração, então uso esse endpoint para diagnóstico, evitando consultas frequentes. Um processo responder ao healthcheck não é, por si só, uma comprovação de que todas as dependências estão disponíveis.

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
