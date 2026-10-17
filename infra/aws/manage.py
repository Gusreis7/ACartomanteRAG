"""Operação na EC2 via SSM; não recebe comandos shell arbitrários."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time

import boto3

ROOT = Path("/srv/acartomante")
CONFIG = Path("/etc/acartomante")
ENV = CONFIG / "runtime.env"


def run(command, *, input=None, capture=False):
    return subprocess.run(
        command, input=input, text=True, check=True, capture_output=capture
    )


def atomic_write(path, data, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(data)
    temp.chmod(mode)
    os.replace(temp, path)


def host_config():
    return json.loads((CONFIG / "host.json").read_text())


def render_environment(image):
    config = host_config()
    settings = json.loads(
        boto3.client("ssm", region_name=config["AWS_REGION"]).get_parameter(
            Name=config["SETTINGS_PARAMETER"],
            WithDecryption=True,
        )["Parameter"]["Value"]
    )
    for key in ("REDIS_PASSWORD", "REDIS_INSPECT_PASSWORD"):
        if not re.fullmatch(r"[0-9a-f]{64}", settings.get(key, "")):
            raise ValueError(f"{key}: use 64 caracteres hexadecimais no SSM.")
    if not settings.get("QDRANT_URL", "").startswith("https://") or not settings.get(
        "QDRANT_API_KEY"
    ):
        raise ValueError("Configure Qdrant Cloud HTTPS e chave de leitura no SSM.")
    values = {
        key: value
        for key, value in config.items()
        if key not in {"SETTINGS_PARAMETER", "ECR_REPOSITORY", "OPS_BUCKET"}
    }
    values.update(
        {
            key: settings[key]
            for key in ("QDRANT_URL", "QDRANT_API_KEY", "REDIS_PASSWORD")
        }
    )
    values.update(
        STORAGE_PROVIDER="s3",
        API_IMAGE=image,
        RUNTIME_ENV_FILE=str(ENV),
        REDIS_URL=f'redis://:{settings["REDIS_PASSWORD"]}@redis:6379/0',
    )
    lines = []
    for key, value in values.items():
        value = str(value)
        if any(char in value for char in "\n\r'$\\"):
            raise ValueError(f"Valor inválido para arquivo de ambiente: {key}")
        lines.append(f"{key}='{value}'")
    atomic_write(ENV, "\n".join(lines) + "\n")
    acl = (
        f'user default on #{hashlib.sha256(settings["REDIS_PASSWORD"].encode()).hexdigest()} ~* &* +@all\n'
        f'user inspector on #{hashlib.sha256(settings["REDIS_INSPECT_PASSWORD"].encode()).hexdigest()} '
        "~* +@read -@dangerous +ping +hello +select +auth +info +scan +command +client|setname +client|id\n"
    )
    atomic_write(ROOT / "data/redis/users.acl", acl, 0o644)
    return config


def compose(release, *arguments, capture=False):
    return run(
        [
            "docker",
            "compose",
            "--env-file",
            str(ENV),
            "-f",
            str(release / "docker-compose.aws.yml"),
            *arguments,
        ],
        capture=capture,
    )


def current_release():
    current = ROOT / "current"
    return current.resolve() if current.exists() else None


def validate_archive(archive, destination):
    allowed = {
        "docker-compose.aws.yml",
        "infra/aws/Caddyfile",
        "infra/aws/manage.py",
        "release.json",
    }
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        if {member.name for member in members} != allowed:
            raise ValueError("Bundle de deploy não corresponde aos arquivos esperados.")
        if any(not member.isfile() or member.size > 1_000_000 for member in members):
            raise ValueError("Tipo ou tamanho de arquivo inválido no bundle.")
        tar.extractall(destination, members=members)


def backup():
    release = current_release()
    if not release:
        return None
    config = host_config()
    if not compose(release, "ps", "-q", "redis", capture=True).stdout.strip():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        # redis-cli --rdb obtém um snapshot consistente pelo protocolo de replicação.
        compose(
            release,
            "exec",
            "-T",
            "redis",
            "sh",
            "-ec",
            'REDISCLI_AUTH="$REDIS_PASSWORD" redis-cli --rdb /tmp/acartomante-backup.rdb >/dev/null',
        )
        container = compose(release, "ps", "-q", "redis", capture=True).stdout.strip()
        target = Path(tmp) / "redis.rdb"
        run(["docker", "cp", f"{container}:/tmp/acartomante-backup.rdb", str(target)])
        stamp = datetime.now(timezone.utc).strftime("%Y/%m/%d/%H%M%S")
        key = f"backups/{stamp}.rdb"
        boto3.client("s3", region_name=config["AWS_REGION"]).upload_file(
            str(target), config["OPS_BUCKET"], key
        )
    print(f"Backup concluído: {key}")
    return key


@contextmanager
def registry_login(config):
    auth = boto3.client(
        "ecr", region_name=config["AWS_REGION"]
    ).get_authorization_token()["authorizationData"][0]
    import base64

    password = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)[1]
    docker_config = tempfile.TemporaryDirectory(prefix="acartomante-docker-")
    old_docker_config = os.environ.get("DOCKER_CONFIG")
    os.environ["DOCKER_CONFIG"] = docker_config.name
    try:
        run(
            [
                "docker",
                "login",
                "--username",
                "AWS",
                "--password-stdin",
                auth["proxyEndpoint"],
            ],
            input=password,
            capture=True,
        )
        yield
    finally:
        if old_docker_config is None:
            os.environ.pop("DOCKER_CONFIG", None)
        else:
            os.environ["DOCKER_CONFIG"] = old_docker_config
        docker_config.cleanup()


def deploy(key, image):
    config = host_config()
    if not re.fullmatch(r"releases/[0-9a-f]{40}/bundle\.tar\.gz", key):
        raise ValueError("Chave de release inválida")
    if not re.fullmatch(
        re.escape(config["ECR_REPOSITORY"]) + r"@sha256:[0-9a-f]{64}", image
    ):
        raise ValueError("Imagem deve pertencer ao ECR configurado e usar digest.")
    sha = key.split("/")[1]
    release = ROOT / "releases" / sha
    release.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=config["AWS_REGION"])
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "bundle.tar.gz"
        s3.download_file(config["OPS_BUCKET"], key, str(archive))
        validate_archive(archive, release)
    if json.loads((release / "release.json").read_text())["git_sha"] != sha:
        raise ValueError("Revisão do bundle não corresponde à chave S3")
    previous = current_release()
    previous_image = (
        json.loads((previous / "deployment.json").read_text())["image"]
        if previous
        else None
    )
    if previous:
        backup()
    render_environment(image)
    with registry_login(config):
        try:
            compose(release, "config", "--quiet")
            compose(release, "pull")
            compose(
                release,
                "up",
                "-d",
                "--remove-orphans",
                "--wait",
                "--wait-timeout",
                "900",
            )
            # Valida o HTTPS real, usando SNI do domínio e o bind privado da EC2.
            run(
                [
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "12",
                    "--retry-delay",
                    "5",
                    "--retry-all-errors",
                    "--max-time",
                    "15",
                    "--resolve",
                    f'{config["APP_DOMAIN"]}:443:{config["EC2_BIND_IP"]}',
                    f'https://{config["APP_DOMAIN"]}/',
                    "-o",
                    "/dev/null",
                ]
            )
        except Exception:
            if previous and previous_image:
                render_environment(previous_image)
                compose(
                    previous,
                    "up",
                    "-d",
                    "--remove-orphans",
                    "--wait",
                    "--wait-timeout",
                    "900",
                )
                print("Deploy falhou; release anterior restaurada.")
            raise
    atomic_write(
        release / "deployment.json", json.dumps({"image": image, "release_key": key})
    )
    link = ROOT / "next"
    link.unlink(missing_ok=True)
    link.symlink_to(release)
    os.replace(link, ROOT / "current")
    atomic_write(
        Path("/opt/acartomante/manage.py"),
        (release / "infra/aws/manage.py").read_text(),
        0o700,
    )
    print(f"Deploy concluído: {sha}")


def restore(key):
    if not re.fullmatch(r"backups/[0-9/]+\.rdb", key):
        raise ValueError("Chave de backup inválida")
    config = host_config()
    release = current_release()
    if not release:
        raise ValueError("Disponibilize uma release para restaurar o Redis.")
    staged = ROOT / "restore-staged"
    staged.mkdir(exist_ok=True)
    target = staged / "dump.rdb"
    boto3.client("s3", region_name=config["AWS_REGION"]).download_file(
        config["OPS_BUCKET"], key, str(target)
    )
    run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{staged}:/restore:ro",
            "--entrypoint",
            "redis-check-rdb",
            "redis/redis-stack-server:7.4.0-v8",
            "/restore/dump.rdb",
        ]
    )
    compose(release, "stop", "api", "redis")
    old = (
        ROOT
        / "data"
        / ("redis.before-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    )
    (ROOT / "data/redis").rename(old)
    (ROOT / "data/redis").mkdir()
    shutil.copy2(target, ROOT / "data/redis/dump.rdb")
    image = json.loads((release / "deployment.json").read_text())["image"]
    render_environment(image)
    # Carrega o RDB com AOF desativado. Depois materializa AOF a partir desses
    # dados antes de voltar à configuração normal, que prefere AOF no startup.
    atomic_write(
        ENV,
        ENV.read_text()
        + "REDIS_ARGS_OVERRIDE='--appendonly no --aclfile /data/users.acl'\n",
    )
    compose(release, "up", "-d", "--wait", "--wait-timeout", "120", "redis")
    compose(
        release,
        "exec",
        "-T",
        "redis",
        "sh",
        "-ec",
        'REDISCLI_AUTH="$REDIS_PASSWORD" redis-cli CONFIG SET appendonly yes',
    )
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        info = compose(
            release,
            "exec",
            "-T",
            "redis",
            "sh",
            "-ec",
            'REDISCLI_AUTH="$REDIS_PASSWORD" redis-cli INFO persistence',
            capture=True,
        ).stdout
        if (
            "aof_rewrite_in_progress:0" in info
            and "aof_last_bgrewrite_status:ok" in info
        ):
            break
        time.sleep(2)
    else:
        raise TimeoutError(
            "RDB carregado, mas a criação do AOF não foi concluída; cópia anterior preservada."
        )
    render_environment(image)
    compose(release, "up", "-d", "--wait", "--wait-timeout", "900")
    print(f"Redis restaurado; dados anteriores preservados em {old}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    d = commands.add_parser("deploy")
    d.add_argument("--release-key", required=True)
    d.add_argument("--image", required=True)
    commands.add_parser("backup")
    commands.add_parser("start")
    commands.add_parser("restart")
    r = commands.add_parser("restore")
    r.add_argument("--key", required=True)
    args = p.parse_args()
    if args.command == "deploy":
        deploy(args.release_key, args.image)
    elif args.command == "backup":
        backup()
    elif args.command == "restore":
        restore(args.key)
    elif args.command == "restart":
        release = current_release()
        if not release:
            raise ValueError("Nenhuma release disponível para reiniciar.")
        compose(release, "restart", "api")
        compose(release, "up", "-d", "--wait", "--wait-timeout", "900", "api")
    elif current_release():
        release = current_release()
        render_environment(
            json.loads((release / "deployment.json").read_text())["image"]
        )
        with registry_login(host_config()):
            compose(release, "pull")
            compose(release, "up", "-d", "--wait", "--wait-timeout", "900")


if __name__ == "__main__":
    with open("/run/lock/acartomante-ops.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
