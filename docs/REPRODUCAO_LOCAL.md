# Rodar no notebook

Este caminho sobe API, Redis, Qdrant e RustFS em Docker, com Ollama no host. Não precisa de AWS, Gitea ou Raspberry. Uma instalação vazia precisa preparar o corpus e os índices antes de atender perguntas; o nome de uma collection, por mais convincente que pareça, não contém seus dados.

## 1. Preparar o ambiente

Instale Git, Make, curl, Python 3.11+, Ollama e Docker com Compose v2 que suporte `include`. Os comandos usam Bash, a raiz do projeto e Docker com sudo.

```bash
git clone https://github.com/Gusreis7/ACartomanteRAG.git
cd ACartomanteRAG
python3 -m venv .venv
source .venv/bin/activate
python scripts/dependencies.py install ci
test -f config/.env || cp config/.env.example config/.env
chmod 600 config/.env
mkdir -p Volumes/rustfs Volumes/qdrant Volumes/redis
pwd -P
python -c 'import secrets; print(secrets.token_hex(32))'
```

Preencha `config/.env`, que fica fora do Git:

| Campo | Preenchimento |
| --- | --- |
| `API_BIND_IP`, `API_PORT` | `127.0.0.1` e `8000`; para acesso pela LAN, use o IP específico do notebook. |
| `SESSION_COOKIE_SECURE` | `false` para HTTP local. |
| `RUSTFS_ACCESS_KEY`, `RUSTFS_SECRET_KEY`, `QDRANT_API_KEY`, `REDIS_PASSWORD` | Gere um valor diferente para cada campo; use hexadecimal na senha Redis. |
| `RUSTFS_VOLUME`, `QDRANT_VOLUME`, `REDIS_VOLUME` | Caminhos absolutos das três pastas criadas. |
| `OLLAMA_BASE_URL_DOCKER` | `http://host.docker.internal:11434`. |

Em uma instalação existente, preserve credenciais e volumes. O perfil `ci` instala os testes na venv; a imagem Docker instala os modelos e as dependências de runtime.

## 2. Conectar o Ollama ao Docker

No Linux, Ollama precisa escutar em uma interface alcançável pela bridge. Se estiver apenas no loopback:

```bash
docker_gateway="$(sudo docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')"
printf '%s\n' "$docker_gateway"
sudo systemctl edit ollama.service
```

No override, substitua `IP_DO_GATEWAY` pelo endereço exibido:

```ini
[Service]
Environment="OLLAMA_HOST=IP_DO_GATEWAY:11434"
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
OLLAMA_HOST="$docker_gateway:11434" ollama pull qwen3.5:4b
```

Permita o acesso da bridge no firewall. O bind deve ser alcançável pelos containers, sem abrir o Ollama para toda a rede.

Mantenha este terminal para reutilizar o atalho Compose:

```bash
dc=(sudo docker compose --env-file config/.env -f docker-compose.local.yml)
"${dc[@]}" config --quiet
"${dc[@]}" up -d redis qdrant rustfs
"${dc[@]}" build tools
"${dc[@]}" run --rm tools python -c 'import os, urllib.request; urllib.request.urlopen(os.environ["OLLAMA_BASE_URL"] + "/api/tags", timeout=10).close(); print("Ollama acessível")'
```

`tools` usa os endereços internos dos backends; suas portas não são publicadas no notebook. Executar os jobs diretamente no host com os endpoints `localhost` do exemplo não substitui essa rede.

## 3. Preparar os dados

Se os volumes já têm corpus e índices compatíveis, pule para a etapa 4. Na primeira instalação:

```bash
"${dc[@]}" run --rm tools python -m src.etl.crawler_ingestao_raw_rustfs \
  --pagina-inicial 1 --pagina-final 7 --delay 2
"${dc[@]}" run --rm tools python -m src.etl.preparar_camada_bronze_rustfs
"${dc[@]}" run --rm tools python -m src.etl.preparar_camada_silver_chunking
```

Confira o resumo e o manifest da coleta antes de continuar. Bronze usa PDFs com status `uploaded` ou `existing`; pendências reduzem o conjunto disponível. `--strict` encerra com código 2 quando a coleta fica incompleta. O corpus inteiro pode levar horas em CPU.

Anote o ID Silver exibido em `datasets/<ID>/` e depois o ID do índice:

```bash
SILVER_ID='ID_GERADO_NA_SILVER'
"${dc[@]}" run --rm tools python -m src.indexing.build_embedding_index \
  "$SILVER_ID" --model colibri
INDEX_ID='ID_GERADO_NO_BUILD'
"${dc[@]}" run --rm tools python -m src.indexing.publish_index \
  --index-id "$INDEX_ID" --release default
```

Em `config/config.yaml`, selecione seus artefatos:

```yaml
retrieval_index:
  index_id: null
  release: default
```

Um `index_id` preenchido tem precedência. Para começar só com recuperação textual, use também `enrichment_retrieval.enabled: false`. Para testar o caminho completo:

```bash
"${dc[@]}" run --rm tools python -m src.enrichment.build_enrichment_dataset --retry
"${dc[@]}" run --rm tools python -m src.indexing.build_enrichment_embedding_indexes \
  --embedders colibri --resume
```

Só construa o índice auxiliar depois de concluir o enrichment. Código 2 indica pendências salvas; `--retry` reaproveita o que terminou. Reative `enrichment_retrieval.enabled` e reconstrua a API. O fallback local `fail_open=true` permite faltar enrichment, mas não valida essa parte do fluxo.

## 4. Subir e testar

```bash
make local DOCKER="sudo docker"
make local-ready DOCKER="sudo docker"
APP_URL="http://$("${dc[@]}" port api 8000)"
curl -fsS "$APP_URL/health"
curl -fsS "$APP_URL/ready"
```

Espere `runtime_ready` nos logs. `/health` retorna `ok`; `/ready`, `ready`. Abra a URL efetiva no navegador e confira:

1. Tutorial, nova conversa e tiragem; depois troque de guia ou escolha neutro.
2. Uma pergunta literária, como “Quem é Capitu?”, com resposta e evidências.
3. Uma pergunta fora do escopo, que deve seguir para a recusa.
4. Outra sessão em janela privada, sem acesso às conversas da primeira.

O teste manual chama o modelo real. Para os testes automatizados:

```bash
make test PYTHON=.venv/bin/python
```

| Durante o desenvolvimento | Comando |
| --- | --- |
| Código ou YAML alterado | `make local DOCKER="sudo docker"` |
| `.env` alterado | `make local-restart DOCKER="sudo docker"` |
| Ver logs ou endereço | `make local-logs` / `make local-url`, com `DOCKER="sudo docker"` |
| Parar e remover containers, preservando dados | `make local-down DOCKER="sudo docker"` |

Não acrescente `-v` ao comando de remoção. Se a URL falhar, confira a porta efetiva; se a prontidão falhar, confira Ollama, manifests e collections. Arquivos salvos apenas dentro de `tools --rm` desaparecem na saída; relatórios precisam de volume ou upload.

O [guia de contribuição](../CONTRIBUTING.md) reúne hooks e PRs. O Raspberry do meu homelab está com o runtime suspenso; ele não faz parte deste procedimento.
