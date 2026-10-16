"""Comandos AWS reproduzíveis; use AWS SSO/profile no notebook e OIDC no CI."""

from __future__ import annotations
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlsplit

import boto3
from botocore.exceptions import ClientError

GLUE_FILES = ("job.py", "source.zip", "dependencies.gluewheels.zip")
ETL_STEPS = (
    "crawl",
    "bronze",
    "silver",
    "text-index",
    "enrichment",
    "enrichment-index",
)


def read_secret_file(path):
    from dotenv import dotenv_values

    if not path.is_file():
        raise ValueError(
            "Arquivo privado não encontrado; copie config/aws-bootstrap.env.example para config/aws-bootstrap.env."
        )
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise ValueError(
            "Arquivo de segredos deve ser privado: chmod 600 no arquivo informado."
        )
    values = dotenv_values(path, interpolate=False)
    required = ("QDRANT_URL", "QDRANT_READ_API_KEY", "QDRANT_WRITE_API_KEY")
    if any(not values.get(key) for key in required):
        raise ValueError(
            "Preencha QDRANT_URL, QDRANT_READ_API_KEY e QDRANT_WRITE_API_KEY no arquivo privado."
        )
    selected = {key: values[key] for key in required}
    validate_secret_values(selected)
    return selected


def validate_secret_values(values):
    url = urlsplit(values["QDRANT_URL"])
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "QDRANT_URL deve ser um endpoint HTTPS sem credenciais, query ou fragmento."
        )
    if any(
        not value or any(char in value for char in "\n\r'$\\")
        for value in values.values()
    ):
        raise ValueError(
            "Endpoint ou chave inválida para a configuração de runtime; revise o arquivo privado."
        )


def configure_secrets(ssm, out, values):
    import secrets

    validate_secret_values(values)
    try:
        old = json.loads(
            ssm.get_parameter(Name=out["app_settings_parameter"], WithDecryption=True)[
                "Parameter"
            ]["Value"]
        )
    except ssm.exceptions.ParameterNotFound:
        old = {}
    passwords = {
        key: old.get(key, secrets.token_hex(32))
        for key in ("REDIS_PASSWORD", "REDIS_INSPECT_PASSWORD")
    }
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in passwords.values()
    ):
        raise ValueError(
            "Senha Redis existente fora do formato esperado; nenhuma rotação automática foi feita."
        )
    app = {
        "QDRANT_URL": values["QDRANT_URL"],
        "QDRANT_API_KEY": values["QDRANT_READ_API_KEY"],
        **passwords,
    }
    etl = {
        "QDRANT_URL": values["QDRANT_URL"],
        "QDRANT_API_KEY": values["QDRANT_WRITE_API_KEY"],
    }
    for name, payload in (
        (out["app_settings_parameter"], app),
        (out["etl_settings_parameter"], etl),
    ):
        ssm.put_parameter(
            Name=name, Type="SecureString", Value=json.dumps(payload), Overwrite=True
        )


def revision_prefix(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use uma revisão Git completa de 40 caracteres.")
    return f"glue/{revision}"


def bundle_keys(s3, bucket, prefix):
    result = s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/", MaxKeys=100)
    if result.get("IsTruncated"):
        raise ValueError(
            "Prefixo de bundle inesperadamente grande; publicação recusada."
        )
    return {item["Key"] for item in result.get("Contents", [])}


def bundle_manifest(s3, bucket, revision):
    prefix = revision_prefix(revision)
    if f"{prefix}/bundle.json" not in bundle_keys(s3, bucket, prefix):
        return None
    manifest = json.loads(
        s3.get_object(Bucket=bucket, Key=f"{prefix}/bundle.json")["Body"].read()
    )
    if manifest.get("revision") != revision or set(manifest.get("files", {})) != set(
        GLUE_FILES
    ):
        raise ValueError("Manifest do bundle diverge da revisão solicitada.")
    for name, expected in manifest["files"].items():
        head = s3.head_object(Bucket=bucket, Key=f"{prefix}/{name}")
        if head["ContentLength"] != expected["size"] or head.get("Metadata", {}) != {
            "revision": revision,
            "sha256": expected["sha256"],
        }:
            raise ValueError(
                "Bundle incompleto ou alterado; publique uma nova revisão."
            )
    return manifest


def publish_glue(s3, bucket, revision, directory):
    prefix = revision_prefix(revision)
    if bundle_manifest(s3, bucket, revision):
        return False
    if any(not (directory / name).is_file() for name in GLUE_FILES):
        raise ValueError(
            "Bundle local incompleto: prepare job.py, source.zip e dependencies.gluewheels.zip."
        )
    known = bundle_keys(s3, bucket, prefix)
    files = {}
    for name in GLUE_FILES:
        path = directory / name
        key = f"{prefix}/{name}"
        if key not in known:
            with path.open("rb") as body:
                digest = hashlib.file_digest(body, "sha256").hexdigest()
            try:
                with path.open("rb") as body:
                    s3.put_object(
                        Bucket=bucket,
                        Key=key,
                        Body=body,
                        Metadata={"revision": revision, "sha256": digest},
                        IfNoneMatch="*",
                    )
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in {"PreconditionFailed", "412"}:
                    raise
        head = s3.head_object(Bucket=bucket, Key=key)
        metadata = head.get("Metadata", {})
        if metadata.get("revision") != revision or not re.fullmatch(
            r"[0-9a-f]{64}", metadata.get("sha256", "")
        ):
            raise ValueError(
                "Artefato já existente sem identidade válida; use outra revisão, sem sobrescrever."
            )
        files[name] = {"sha256": metadata["sha256"], "size": head["ContentLength"]}
    manifest = {"revision": revision, "files": files}
    try:
        s3.put_object(
            Bucket=bucket,
            Key=f"{prefix}/bundle.json",
            Body=json.dumps(manifest, sort_keys=True).encode(),
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"PreconditionFailed", "412"}:
            raise
    if bundle_manifest(s3, bucket, revision) != manifest:
        raise ValueError(
            "Manifest concorrente diverge; nenhum artefato foi sobrescrito."
        )
    return True


def glue_arguments(
    revision, bucket, run_id, steps, promote=False, smoke=False, raw_id=""
):
    prefix = revision_prefix(revision)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        raise ValueError(
            "run-id inválido; use letras, números, hífen ou sublinhado, até 100 caracteres."
        )
    selected = steps.split(",")
    if any(step not in ETL_STEPS for step in selected) or selected != sorted(
        set(selected), key=ETL_STEPS.index
    ):
        raise ValueError("Etapas inválidas ou fora da ordem do pipeline.")
    if smoke and promote:
        raise ValueError("Smoke não promove dados.")
    if raw_id and (
        not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", raw_id)
        or tuple(selected) != ETL_STEPS[1:]
        or smoke
    ):
        raise ValueError(
            "raw-id exige somente bronze, silver, text-index, enrichment e enrichment-index."
        )
    arguments = {
        "--scriptLocation": f"s3://{bucket}/{prefix}/job.py",
        "--additional-python-modules": f"s3://{bucket}/{prefix}/dependencies.gluewheels.zip",
        "--bundle-prefix": prefix,
        "--run-id": run_id,
        "--steps": steps,
        "--promote": str(promote).lower(),
        "--smoke": str(smoke).lower(),
    }
    if raw_id:
        arguments["--raw-id"] = raw_id
    return arguments


def outputs(path):
    if path:
        value = json.loads(path.read_text())
    else:
        value = json.loads(
            subprocess.run(
                ["terraform", "-chdir=infra/aws", "output", "-json"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout
        )
    return {key: item["value"] for key, item in value.items()}


def wait_command(client, command_id, instance):
    deadline = time.monotonic() + 1900
    while time.monotonic() < deadline:
        try:
            result = client.get_command_invocation(
                CommandId=command_id, InstanceId=instance
            )
        except client.exceptions.InvocationDoesNotExist:
            time.sleep(5)
            continue
        if result["Status"] in {"Success", "Failed", "TimedOut", "Cancelled"}:
            print(result.get("StandardOutputContent", "").strip())
            if result["Status"] != "Success":
                raise RuntimeError(
                    f'SSM {result["Status"]}: {result.get("StandardErrorContent", "")[-3000:]}'
                )
            return
        time.sleep(10)
    raise TimeoutError(
        f"Timeout aguardando comando {command_id}; confira seu estado antes de repetir."
    )


def send_when_ready(client, **request):
    deadline = time.monotonic() + 600
    announced = False
    while True:
        try:
            return client.send_command(**request)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {
                "InvalidInstanceId",
                "InvalidInstanceIdException",
                "InvalidInstance",
                "TargetNotConnected",
            }:
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "O host ainda não está disponível no SSM; confira cloud-init e a role da EC2."
                ) from exc
            if not announced:
                print("Aguardando o host ficar disponível no SSM.")
                announced = True
            time.sleep(10)


def wait_job(client, job, run_id):
    previous = None
    while True:
        run = client.get_job_run(JobName=job, RunId=run_id)["JobRun"]
        status = run["JobRunState"]
        if status != previous:
            print(f"Glue: {status}")
            previous = status
        if status == "SUCCEEDED":
            return
        if status in {"FAILED", "ERROR", "TIMEOUT", "STOPPED", "EXPIRED"}:
            raise RuntimeError(
                f'Glue {status}: {run.get("ErrorMessage", "Consulte CloudWatch")[:500]}'
            )
        time.sleep(15)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", type=Path, help="JSON de terraform output -json")
    commands = p.add_subparsers(dest="command", required=True)
    d = commands.add_parser("deploy")
    d.add_argument("--instance")
    d.add_argument("--document")
    d.add_argument("--release", required=True)
    d.add_argument("--image", required=True)
    secret = commands.add_parser("secrets")
    secret.add_argument("--env-file", type=Path)
    secret.add_argument(
        "--check",
        action="store_true",
        help="Valida o arquivo sem consultar ou alterar a AWS",
    )
    g = commands.add_parser("glue")
    g.add_argument("--revision", required=True)
    g.add_argument("--run-id", required=True)
    g.add_argument("--steps", default=",".join(ETL_STEPS))
    g.add_argument("--reuse-index-id", default="")
    g.add_argument("--raw-id", default="")
    g.add_argument("--promote", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--wait", action="store_true")
    u = commands.add_parser("upload-glue")
    u.add_argument("--revision", required=True)
    u.add_argument("--directory", type=Path, default=Path("dist/glue"))
    check = commands.add_parser("check-glue")
    check.add_argument("--revision", required=True)
    reload = commands.add_parser("reload-data")
    reload.add_argument("--instance")
    reload.add_argument("--document")
    commands.add_parser("redis-tunnel")
    commands.add_parser("redis-password")
    a = commands.add_parser("admin")
    a.add_argument("operation", choices=["backup", "restart", "restore"])
    a.add_argument("--key")
    r = commands.add_parser("data-rollback")
    r.add_argument("--run-id", required=True)
    args = p.parse_args()
    if args.command == "secrets" and args.check:
        if not args.env_file:
            p.error("--check exige --env-file")
        read_secret_file(args.env_file)
        print("Arquivo privado válido; nenhum segredo exibido ou enviado.")
        return
    region = os.getenv("AWS_REGION", "us-east-1")
    ssm = boto3.client("ssm", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    if args.command in {"deploy", "reload-data"}:
        target = outputs(args.outputs) if args.outputs else {}
        instance = args.instance or target.get("instance_id")
        document = args.document or target.get(
            "ssm_deploy_document"
            if args.command == "deploy"
            else "ssm_data_reload_document"
        )
        if not instance or not document:
            p.error(
                "Informe --outputs com o contexto atual, ou --instance e --document."
            )
        parameters = {}
        if args.command == "deploy":
            if not re.fullmatch(r"[0-9a-f]{40}", args.release):
                raise ValueError("Revisão Git inválida")
            parameters = {
                "ReleaseKey": [f"releases/{args.release}/bundle.tar.gz"],
                "Image": [args.image],
            }
        command = send_when_ready(
            ssm, InstanceIds=[instance], DocumentName=document, Parameters=parameters
        )["Command"]["CommandId"]
        wait_command(ssm, command, instance)
        return
    out = outputs(args.outputs)
    if args.command == "secrets":
        values = (
            read_secret_file(args.env_file)
            if args.env_file
            else {
                "QDRANT_URL": input("Qdrant Cloud URL HTTPS: ").strip(),
                "QDRANT_READ_API_KEY": getpass.getpass(
                    "Qdrant API key SOMENTE LEITURA: "
                ),
                "QDRANT_WRITE_API_KEY": getpass.getpass(
                    "Qdrant API key ESCRITA (ETL): "
                ),
            }
        )
        configure_secrets(ssm, out, values)
        print("Segredos configurados no SSM; senhas Redis existentes preservadas.")
    elif args.command == "upload-glue":
        created = publish_glue(s3, out["ops_bucket"], args.revision, args.directory)
        print(
            "Bundle Glue publicado; nenhum job executado."
            if created
            else "Bundle Glue íntegro reutilizado; nenhum job executado."
        )
    elif args.command == "check-glue":
        if not bundle_manifest(s3, out["ops_bucket"], args.revision):
            print("Bundle Glue ainda não publicado.")
            raise SystemExit(3)
        print("Bundle Glue íntegro e pronto.")
    elif args.command == "glue":
        arguments = glue_arguments(
            args.revision,
            out["ops_bucket"],
            args.run_id,
            args.steps,
            args.promote,
            args.smoke,
            args.raw_id,
        )
        if args.raw_id and args.reuse_index_id:
            raise ValueError(
                "O processamento a partir do Raw gera um novo índice; não use reuse-index-id."
            )
        if not bundle_manifest(s3, out["ops_bucket"], args.revision):
            raise ValueError("Publique o bundle Glue completo antes de iniciar o job.")
        client = boto3.client("glue", region_name=region)
        if args.reuse_index_id:
            arguments["--reuse-index-id"] = args.reuse_index_id
        run_id = client.start_job_run(JobName=out["glue_job"], Arguments=arguments)[
            "JobRunId"
        ]
        print(f"Glue iniciado: {run_id}")
        if args.wait:
            wait_job(client, out["glue_job"], run_id)
    elif args.command == "redis-tunnel":
        subprocess.run(
            [
                "aws",
                "ssm",
                "start-session",
                "--region",
                region,
                "--target",
                out["instance_id"],
                "--document-name",
                "AWS-StartPortForwardingSession",
                "--parameters",
                json.dumps({"portNumber": ["6379"], "localPortNumber": ["16379"]}),
            ],
            check=True,
        )
    elif args.command == "redis-password":
        settings = json.loads(
            ssm.get_parameter(Name=out["app_settings_parameter"], WithDecryption=True)[
                "Parameter"
            ]["Value"]
        )
        print(settings["REDIS_INSPECT_PASSWORD"])
    elif args.command == "admin":
        command = f"python3 /opt/acartomante/manage.py {args.operation}"
        if args.operation == "restore":
            if not args.key or not re.fullmatch(r"backups/[0-9/]+\.rdb", args.key):
                raise ValueError("Informe uma chave backups/...rdb")
            command += f" --key {args.key}"
        invocation = ssm.send_command(
            InstanceIds=[out["instance_id"]],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [command], "executionTimeout": ["1800"]},
        )["Command"]["CommandId"]
        wait_command(ssm, invocation, out["instance_id"])
    elif args.command == "data-rollback":
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", args.run_id):
            raise ValueError("run-id inválido")
        bucket = out["buckets"]["EMBEDDING_INDEXES_BUCKET"]
        key = f"production/releases/{args.run_id}.json"
        release = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        if release.get("status") != "success":
            raise ValueError("Release não validada")
        s3.copy_object(
            Bucket=bucket,
            Key="production/current.json",
            CopySource={"Bucket": bucket, "Key": key},
        )
        print("Ponteiro restaurado. Execute admin restart para recarregar os dados.")


if __name__ == "__main__":
    main()
