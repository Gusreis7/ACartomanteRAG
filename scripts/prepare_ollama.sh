#!/bin/sh
set -eu
: "${OLLAMA_MODEL:?Defina OLLAMA_MODEL}"
: "${OLLAMA_HOST:?Defina OLLAMA_HOST}"

if ollama show "$OLLAMA_MODEL" >/dev/null 2>&1; then
    printf '[ollama] Modelo disponível: %s\n' "$OLLAMA_MODEL"
    exit 0
fi

printf '[ollama] Preparando modelo: %s. O primeiro download pode demorar.\n' "$OLLAMA_MODEL"
pull_log=$(mktemp /tmp/acartomante-ollama-pull.XXXXXX)
trap 'rm -f "$pull_log"' EXIT HUP INT TERM
if ! ollama pull "$OLLAMA_MODEL" >"$pull_log" 2>&1; then
    printf '[ollama] Falha ao baixar %s; a API não será iniciada.\n' "$OLLAMA_MODEL" >&2
    tail -n 20 "$pull_log" >&2
    exit 1
fi
if ! ollama show "$OLLAMA_MODEL" >/dev/null 2>&1; then
    printf '[ollama] O modelo %s não ficou disponível após o download.\n' "$OLLAMA_MODEL" >&2
    exit 1
fi
printf '[ollama] Modelo pronto: %s\n' "$OLLAMA_MODEL"
