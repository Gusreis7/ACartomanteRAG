#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ "${COMPOSE_PROJECT_NAME:-}" == acartomante-ci-* ]] || {
  printf '%s\n' 'Erro: esta checagem exige um projeto temporário acartomante-ci-*.' >&2
  exit 1
}
compose=(bash "$PROJECT_ROOT/scripts/compose_raspberry.sh")
"${compose[@]}" config --quiet
"${compose[@]}" build api
# O CI tem volumes novos, sem os manifests e collections do corpus publicado.
# Testa os backends e o servidor Ollama; não baixa 4 GB nem inicializa o RAG.
"${compose[@]}" up --detach --wait --wait-timeout 240 redis qdrant rustfs ollama
"${compose[@]}" run --rm --no-deps \
  -e AWS_EC2_METADATA_DISABLED=true --entrypoint python api \
  -c 'import src.api.main; import src.generation.provider; import src.retrieval.enrichment'
printf '%s\n' 'Infraestrutura local e imports da API validados; o RAG completo exige o corpus publicado.'
