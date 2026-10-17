"""Instala/exporta perfis do pyproject.toml sem depender de bibliotecas externas."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def load(path=ROOT / "pyproject.toml"):
    with path.open("rb") as source:
        data = tomllib.load(source)
    groups = {}
    for name, entries in data["dependency-groups"].items():
        key = normalize(name)
        if key in groups:
            raise ValueError("Perfis de dependências com nomes equivalentes.")
        groups[key] = entries
    return groups, data["tool"]["acartomante"]["dependencies"]


def resolve(groups, name, ancestors=()):
    key = normalize(name)
    if key in ancestors:
        raise ValueError("Ciclo entre perfis de dependências.")
    if key not in groups or not isinstance(groups[key], list):
        raise ValueError(f"Perfil de dependências inválido: {name}")
    result = []
    for item in groups[key]:
        if isinstance(item, str) and re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*", item) and not any(char in item for char in "\r\n"):
            result.append(item)
        elif isinstance(item, dict) and set(item) == {"include-group"} and isinstance(item["include-group"], str):
            result.extend(resolve(groups, item["include-group"], (*ancestors, key)))
        else:
            raise ValueError(f"Entrada inválida no perfil {name}.")
    return result


def constraints(groups, settings):
    lines = resolve(groups, settings["constraints-group"])
    if not all(re.match(r"^[A-Za-z0-9_.-]+==[^\s;]+(?:\s*;.+)?$", line) for line in lines):
        raise ValueError("O grupo de constraints deve conter somente versões exatas, sem extras.")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Valida todos os perfis sem instalar")
    export = commands.add_parser("export", help="Gera requirements temporário para pip/wheels")
    export.add_argument("profile")
    export.add_argument("--output", type=Path)
    install = commands.add_parser("install", help="Instala o perfil e aplica os pins centrais")
    install.add_argument("profile")
    install.add_argument("--no-cache-dir", action="store_true")
    args = parser.parse_args()
    groups, settings = load(args.project / "pyproject.toml")
    locked = constraints(groups, settings)
    index = settings["cpu-index-url"]
    if index != "https://download.pytorch.org/whl/cpu":
        raise ValueError("Use o índice CPU oficial configurado no manifesto.")
    if args.command == "check":
        for name in groups:
            resolve(groups, name)
        print(f"Dependências válidas: {', '.join(groups)}.")
        return
    requested = resolve(groups, args.profile)
    # Perfis leves não precisam consultar o repositório de wheels PyTorch.
    cpu = any(re.match(r"^torch(?:\[|[<>=!~;\s]|$)", line, re.IGNORECASE) for line in requested)
    if args.command == "export":
        payload = ("--extra-index-url " + index + "\n" if cpu else "") + "\n".join(requested) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return
    with tempfile.TemporaryDirectory(prefix="acartomante-deps-") as directory:
        requirement_file = Path(directory) / "requirements.txt"
        constraint_file = Path(directory) / "constraints.txt"
        requirement_file.write_text("\n".join(requested) + "\n", encoding="utf-8")
        constraint_file.write_text("\n".join(locked) + "\n", encoding="utf-8")
        command = [sys.executable, "-m", "pip", "install"]
        if cpu:
            command += ["--extra-index-url", index]
        command += ["--constraint", str(constraint_file), "--requirement", str(requirement_file)]
        if args.no_cache_dir:
            command.append("--no-cache-dir")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
