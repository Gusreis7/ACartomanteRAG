#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
command -v docker >/dev/null 2>&1 || { printf '%s\n' 'Erro: Docker não encontrado.' >&2; exit 1; }
[[ $# -gt 0 ]] || { printf '%s\n' 'Uso: bash scripts/compose_raspberry.sh <comando compose> [opções]' >&2; exit 2; }

# O runner injeta variáveis do job. No host, uso o .env privado existente.
options=(--project-directory "$PROJECT_ROOT")
if [[ -n "${LOCAL_ENV_FILE:-}" ]]; then
  [[ -f "$LOCAL_ENV_FILE" ]] || { printf '%s\n' 'Erro: LOCAL_ENV_FILE não encontrado.' >&2; exit 1; }
  options+=(--env-file "$LOCAL_ENV_FILE")
elif [[ -f "$PROJECT_ROOT/config/.env" ]]; then
  options+=(--env-file "$PROJECT_ROOT/config/.env")
fi
if [[ "${COMPOSE_PROJECT_NAME:-}" == acartomante-ci-* ]]; then
  export API_BIND_IP="${API_BIND_IP:-127.0.0.1}"
fi
exec docker compose "${options[@]}" \
  -f "$PROJECT_ROOT/docker-compose.yml" \
  -f "$PROJECT_ROOT/docker-compose.raspberry.yml" "$@"
