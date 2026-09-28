#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BRANCH="${GITEA_REF_NAME:-${GITHUB_REF_NAME:-${CI_COMMIT_BRANCH:-${BRANCH_NAME:-}}}}"

case "${BRANCH}" in
  dev|develop)
    RELEASE="${ACARTOMANTE_DEV_RELEASE:-dev}"
    ;;
  main|master)
    RELEASE="${ACARTOMANTE_MAIN_RELEASE:-default}"
    ;;
  *)
    echo "[skip] Branch '${BRANCH}' não promove release."
    exit 0
    ;;
esac

PINNED_INDEX="$(
  python - <<'PYCFG'
from pathlib import Path
import yaml

cfg = (
    yaml.safe_load(
        Path("config/config.yaml").read_text(
            encoding="utf-8"
        )
    )
    or {}
)

value = (
    cfg.get("retrieval_index")
    or {}
).get("index_id")

print(
    ""
    if value is None
    else str(value).strip()
)
PYCFG
)"

if [[ -n "${PINNED_INDEX}" ]]; then
  echo "[skip] retrieval_index.index_id pinado em '${PINNED_INDEX}'."
  echo "[skip] A release '${RELEASE}' não será alterada."
  exit 0
fi

if [[ -z "${INDEX_ID:-}" ]]; then
  echo "INDEX_ID não definido. Assumindo default"
  INDEX_ID="default"
fi

echo "[+] Branch:  ${BRANCH}"
echo "[+] Release: ${RELEASE}"
echo "[+] Index:   ${INDEX_ID}"

python -m src.indexing.publish_index   --index-id "${INDEX_ID}"   --release "${RELEASE}"

echo "[ok] Release promovida: ${RELEASE} -> ${INDEX_ID}"
