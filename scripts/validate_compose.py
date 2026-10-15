"""Valida os Composes local, Raspberry e AWS; não sobe containers."""

import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    values = {
        "API_BIND_IP": "192.0.2.10",
        "RUSTFS_ACCESS_KEY": "test",
        "RUSTFS_SECRET_KEY": "test",
        "QDRANT_API_KEY": "test",
        "REDIS_PASSWORD": "a" * 64,
        "REDIS_VOLUME": "/tmp/acartomante-redis",
        "RUSTFS_VOLUME": "/tmp/acartomante-rustfs",
        "QDRANT_VOLUME": "/tmp/acartomante-qdrant",
        "APP_DOMAIN": "acartomante.example.org",
        "ACME_EMAIL": "test@example.org",
        "EC2_BIND_IP": "10.42.1.10",
        "API_IMAGE": "example.org/acartomante@sha256:" + "a" * 64,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env") as temp:
        temp.write("\n".join(f"{k}={v}" for k, v in values.items()))
        temp.flush()
        values["RUNTIME_ENV_FILE"] = temp.name
        env = {**os.environ, **values}
        for name in ("docker-compose.local.yml", "docker-compose.aws.yml"):
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "--env-file",
                    temp.name,
                    "-f",
                    str(ROOT / name),
                    "config",
                    "--quiet",
                ],
                env=env,
                check=True,
            )
        subprocess.run(
            [
                "docker",
                "compose",
                "--env-file",
                temp.name,
                "--profile",
                "tools",
                "-f",
                str(ROOT / "docker-compose.yml"),
                "-f",
                str(ROOT / "docker-compose.raspberry.yml"),
                "config",
                "--quiet",
            ],
            env=env,
            check=True,
        )
    print("Composes válidos")


if __name__ == "__main__":
    main()
