#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Manutenção do Raspberry: somente análise estática, sem importar a aplicação,
# criar imagens, baixar modelos ou tocar no estado dos serviços existentes.
python3 - "$PROJECT_ROOT" <<'PY'
from pathlib import Path
import ast
import subprocess
import sys

root = Path(sys.argv[1])
for directory in ("src", "scripts", "infra", "tests"):
    for path in sorted((root / directory).rglob("*.py")):
        ast.parse(path.read_bytes(), filename=str(path))
for directory in ("scripts", ".gitea/scripts"):
    for path in sorted((root / directory).rglob("*.sh")):
        subprocess.run(["bash", "-n", str(path)], check=True)
PY
python3 "$PROJECT_ROOT/scripts/validate_compose.py"
printf '%s\n' 'Checagens estáticas concluídas; runtime e deploy do Raspberry suspensos. Consulte docs/IMPEDITIVOS.md.'
