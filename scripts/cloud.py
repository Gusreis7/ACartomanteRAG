"""Comandos AWS reproduzíveis; use AWS SSO/profile no notebook e OIDC no CI."""
from __future__ import annotations
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import time

import boto3


def outputs(path):
    if path:
        value = json.loads(path.read_text())
    else:
        value = json.loads(subprocess.run(["terraform", "-chdir=infra/aws", "output", "-json"], text=True, capture_output=True, check=True).stdout)
    return {key: item["value"] for key, item in value.items()}


def wait_command(client, command_id, instance):
    deadline = time.monotonic() + 1900
    while time.monotonic() < deadline:
        try:
            result = client.get_command_invocation(CommandId=command_id, InstanceId=instance)
        except client.exceptions.InvocationDoesNotExist:
            time.sleep(5)
            continue
        if result["Status"] in {"Success", "Failed", "TimedOut", "Cancelled"}:
            print(result.get("StandardOutputContent", "").strip())
            if result["Status"] != "Success":
                raise RuntimeError(f'SSM {result["Status"]}: {result.get("StandardErrorContent", "")[-3000:]}')
            return
        time.sleep(10)
    raise TimeoutError(f"Timeout aguardando comando {command_id}; confira seu estado antes de repetir.")


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
            raise RuntimeError(f'Glue {status}: {run.get("ErrorMessage", "Consulte CloudWatch")[:500]}')
        time.sleep(15)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", type=Path, help="JSON de terraform output -json")
    commands = p.add_subparsers(dest="command", required=True)
    d = commands.add_parser("deploy")
    d.add_argument("--instance", required=True)
    d.add_argument("--document", required=True)
    d.add_argument("--release", required=True)
    d.add_argument("--image", required=True)
    commands.add_parser("secrets")
    g = commands.add_parser("glue")
    g.add_argument("--revision", required=True)
    g.add_argument("--run-id", required=True)
    g.add_argument("--steps", default="crawl,bronze,silver,text-index")
    g.add_argument("--reuse-index-id", default="")
    g.add_argument("--promote", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--wait", action="store_true")
    u = commands.add_parser("upload-glue")
    u.add_argument("--revision", required=True)
    u.add_argument("--directory", type=Path, default=Path("dist/glue"))
    commands.add_parser("redis-tunnel")
    commands.add_parser("redis-password")
    a = commands.add_parser("admin")
    a.add_argument("operation", choices=["backup", "restart", "restore"])
    a.add_argument("--key")
    r = commands.add_parser("data-rollback")
    r.add_argument("--run-id", required=True)
    args = p.parse_args()
    region = os.getenv("AWS_REGION", "us-east-1")
    ssm = boto3.client("ssm", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    if args.command == "deploy":
        if not re.fullmatch(r"[0-9a-f]{40}", args.release):
            raise ValueError("Revisão Git inválida")
        command = ssm.send_command(InstanceIds=[args.instance], DocumentName=args.document, Parameters={
            "ReleaseKey": [f"releases/{args.release}/bundle.tar.gz"], "Image": [args.image],
        })["Command"]["CommandId"]
        wait_command(ssm, command, args.instance)
        return
    out = outputs(args.outputs)
    if args.command == "secrets":
        import secrets
        url = input("Qdrant Cloud URL HTTPS: ").strip()
        read_key = getpass.getpass("Qdrant API key SOMENTE LEITURA: ")
        write_key = getpass.getpass("Qdrant API key ESCRITA (ETL): ")
        if not url.startswith("https://") or not read_key or not write_key:
            raise ValueError("Informe URL HTTPS e ambas as chaves")
        # Preserva senhas existentes ao reconfigurar Qdrant.
        try:
            old = json.loads(ssm.get_parameter(Name=out["app_settings_parameter"], WithDecryption=True)["Parameter"]["Value"])
        except ssm.exceptions.ParameterNotFound:
            old = {}
        app = {"QDRANT_URL": url, "QDRANT_API_KEY": read_key,
               "REDIS_PASSWORD": old.get("REDIS_PASSWORD", secrets.token_hex(32)),
               "REDIS_INSPECT_PASSWORD": old.get("REDIS_INSPECT_PASSWORD", secrets.token_hex(32))}
        etl = {"QDRANT_URL": url, "QDRANT_API_KEY": write_key}
        for name, payload in ((out["app_settings_parameter"], app), (out["etl_settings_parameter"], etl)):
            ssm.put_parameter(Name=name, Type="SecureString", Value=json.dumps(payload), Overwrite=True)
        print("Segredos configurados no SSM; senhas Redis existentes preservadas.")
    elif args.command == "upload-glue":
        if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
            raise ValueError("Revisão Git inválida")
        for name in ("job.py", "source.zip", "dependencies.gluewheels.zip"):
            file = args.directory / name
            if not file.is_file():
                raise FileNotFoundError(file)
        for name in ("job.py", "source.zip", "dependencies.gluewheels.zip"):
            s3.upload_file(str(args.directory / name), out["ops_bucket"], f"glue/{args.revision}/{name}")
        print("Bundle Glue publicado; nenhum job foi executado.")
    elif args.command == "glue":
        if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
            raise ValueError("Revisão Git inválida")
        prefix = f"glue/{args.revision}"
        client = boto3.client("glue", region_name=region)
        job = client.get_job(JobName=out["glue_job"])["Job"]
        expected = f's3://{out["ops_bucket"]}/{prefix}/job.py'
        if job["Command"]["ScriptLocation"] != expected:
            raise ValueError(f"Atualize glue_bundle_prefix={prefix} pelo Terraform antes de rodar esta revisão.")
        arguments = {
            "--run-id": args.run_id, "--steps": args.steps, "--reuse-index-id": args.reuse_index_id,
            "--promote": str(args.promote).lower(), "--smoke": str(args.smoke).lower(),
        }
        run_id = client.start_job_run(JobName=out["glue_job"], Arguments=arguments)["JobRunId"]
        print(f"Glue iniciado: {run_id}")
        if args.wait:
            wait_job(client, out["glue_job"], run_id)
    elif args.command == "redis-tunnel":
        subprocess.run(["aws", "ssm", "start-session", "--region", region, "--target", out["instance_id"],
                        "--document-name", "AWS-StartPortForwardingSession", "--parameters",
                        json.dumps({"portNumber": ["6379"], "localPortNumber": ["16379"]})], check=True)
    elif args.command == "redis-password":
        settings = json.loads(ssm.get_parameter(Name=out["app_settings_parameter"], WithDecryption=True)["Parameter"]["Value"])
        print(settings["REDIS_INSPECT_PASSWORD"])
    elif args.command == "admin":
        command = f"python3 /opt/acartomante/manage.py {args.operation}"
        if args.operation == "restore":
            if not args.key or not re.fullmatch(r"backups/[0-9/]+\.rdb", args.key):
                raise ValueError("Informe uma chave backups/...rdb")
            command += f" --key {args.key}"
        invocation = ssm.send_command(InstanceIds=[out["instance_id"]], DocumentName="AWS-RunShellScript",
                                      Parameters={"commands": [command], "executionTimeout": ["1800"]})["Command"]["CommandId"]
        wait_command(ssm, invocation, out["instance_id"])
    elif args.command == "data-rollback":
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", args.run_id):
            raise ValueError("run-id inválido")
        bucket = out["buckets"]["EMBEDDING_INDEXES_BUCKET"]
        key = f"production/releases/{args.run_id}.json"
        release = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        if release.get("status") != "success":
            raise ValueError("Release não validada")
        s3.copy_object(Bucket=bucket, Key="production/current.json", CopySource={"Bucket": bucket, "Key": key})
        print("Ponteiro restaurado. Execute admin restart para recarregar os dados.")


if __name__ == "__main__":
    main()
