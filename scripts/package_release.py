"""Empacota apenas arquivos necessários ao deploy e aos jobs; sem .env/dados."""

import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sha", required=True)
    p.add_argument("--output", type=Path, default=ROOT / "dist")
    p.add_argument(
        "--source-root",
        type=Path,
        default=ROOT,
        help="Checkout ou git archive da revisão que será empacotada",
    )
    p.add_argument("--only", choices=("all", "app", "glue"), default="all")
    args = p.parse_args()
    root = args.source_root.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        p.error("Use uma revisão completa do Git")
    git = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True
    )
    if git.returncode == 0:
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        if git.stdout.strip() != args.sha or dirty.strip():
            raise ValueError(
                "Versione as alterações antes de empacotar e use o SHA do HEAD limpo."
            )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.only != "glue":
        with tarfile.open(args.output / "bundle.tar.gz", "w:gz") as tar:
            for name in (
                "docker-compose.aws.yml",
                "infra/aws/Caddyfile",
                "infra/aws/manage.py",
            ):
                tar.add(root / name, arcname=name, recursive=False)
            data = json.dumps({"git_sha": args.sha}).encode()
            info = tarfile.TarInfo("release.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    if args.only == "app":
        print(f"Release da aplicação preparada: {args.sha}")
        return
    glue = args.output / "glue"
    glue.mkdir(exist_ok=True)
    with zipfile.ZipFile(
        glue / "source.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as z:
        z.writestr("config/source_revision.txt", args.sha)
        for folder in ("src", "config"):
            for file in sorted((root / folder).rglob("*")):
                if not file.is_file() or "__pycache__" in file.parts:
                    continue
                if file.suffix == ".py" or file.name in {
                    "config.yaml",
                    "config.aws.yaml",
                }:
                    z.write(file, file.relative_to(root))
    (glue / "job.py").write_bytes((root / "infra/glue/job.py").read_bytes())
    print(f"Release e fontes Glue preparadas: {args.sha}")


if __name__ == "__main__":
    main()
