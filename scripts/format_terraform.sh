#!/usr/bin/env bash
set -euo pipefail

options=()
if [[ "${1:-}" == --check ]]; then
  options=(-check -diff)
  shift
fi
[[ $# -gt 0 ]] || exit 0
command -v terraform >/dev/null || {
  printf '%s\n' 'Erro: instale Terraform 1.11.4 para formatar os arquivos .tf deste commit.' >&2
  exit 1
}
# terraform fmt recebe um único arquivo por chamada. Não percorra diretórios:
# o hook deve respeitar os arquivos selecionados e preservar edições fora deles.
for path in "$@"; do
  [[ -f "$path" ]] || continue
  [[ "$path" == /* ]] || path="./$path"
  terraform fmt "${options[@]}" "$path"
done
