"""Bootstrap local: plano revisável, aplicação e configuração inicial do GitHub/SSM."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import boto3

ROOT = Path(__file__).resolve().parents[1]
TF = ROOT / "infra/aws"
PLAN = TF / "bootstrap.tfplan"
RECORD = TF / "bootstrap-plan.local.json"
OUTPUTS = TF / "outputs.local.json"


def run(*arguments, capture=False):
    return subprocess.run([str(argument) for argument in arguments], cwd=ROOT, check=True, text=True, capture_output=capture)


def server_policy(account, project):
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "iam:*", "Resource": [
            *[f"arn:aws:iam::{account}:role/{project}-{suffix}" for suffix in ("runtime", "etl", "github-deploy", "github-infra", "github-etl")],
            f"arn:aws:iam::{account}:instance-profile/{project}",
            f"arn:aws:iam::{account}:oidc-provider/token.actions.githubusercontent.com",
        ]},
        {"Effect": "Allow", "Action": ["iam:ListInstanceProfiles", "iam:ListOpenIDConnectProviders"], "Resource": "*"},
        {"Effect": "Allow", "Action": ["iam:GetPolicy", "iam:GetPolicyVersion"], "Resource": [
            "arn:aws:iam::aws:policy/PowerUserAccess", "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
        ]},
    ]}


def fingerprint(config_path):
    paths = [*sorted(TF.glob("*.tf")), TF / "cloud-init.sh.tftpl", TF / "manage.py", TF / ".terraform.lock.hcl", config_path]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_config(path, policy_only=False):
    if not path.is_file():
        if policy_only:
            return {"project": "acartomanterag", "region": "us-east-1"}
        raise ValueError("Copie infra/aws/production.tfvars.json.example para production.tfvars.json e preencha os campos públicos.")
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("A configuração de produção deve ser um objeto JSON.")
    if any(re.search(r"secret|password|token|api_key|access_key", key, re.I) for key in config):
        raise ValueError("Configuração pública não pode conter segredos; use config/aws-bootstrap.env.")
    declared = set(re.findall(r'variable\s+"([^"\n]+)"', (TF / "variables.tf").read_text()))
    if set(config) - declared:
        raise ValueError("Configuração pública contém campos que não são variáveis Terraform.")
    config.setdefault("project", "acartomanterag")
    config.setdefault("region", "us-east-1")
    if not re.fullmatch(r"[a-z][a-z0-9-]{2,25}", config["project"]):
        raise ValueError("Nome de projeto inválido.")
    if not policy_only:
        if not re.fullmatch(r"[1-9][0-9]*", str(config.get("github_repository_id", ""))):
            raise ValueError("Preencha github_repository_id com o ID numérico real.")
        if any(not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", str(config.get(key, ""))) for key in ("acme_email", "budget_email")):
            raise ValueError("Preencha acme_email e budget_email na configuração pública.")
    return config


def validate_record(record, account, repo, current_fingerprint):
    if record != {"account": account, "repo": repo, "fingerprint": current_fingerprint}:
        raise ValueError("Conta, repositório ou configuração mudou após o plano. Execute plan novamente antes de aplicar.")


def check_legacy_inputs(config_path):
    for path in TF.iterdir():
        if path.resolve() != config_path and (path.name in {"terraform.tfvars", "terraform.tfvars.json"} or path.name.endswith((".auto.tfvars", ".auto.tfvars.json"))):
            raise ValueError("Há tfvars carregados automaticamente no diretório. Transfira suas escolhas para production.tfvars.json e guarde os legados fora do checkout, para usar a mesma configuração no GitHub.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("policy", "plan", "apply", "configure"))
    parser.add_argument("--repo", help="owner/repository no GitHub")
    parser.add_argument("--env-file", type=Path, default=ROOT / "config/aws-bootstrap.env")
    parser.add_argument("--config", type=Path, default=TF / "production.tfvars.json")
    parser.add_argument("--protect-main", action="store_true", help="Exige PR no GitHub; não usar para um espelho de push do Gitea")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.env_file = args.env_file.resolve()
    config = load_config(args.config, policy_only=args.operation == "policy")
    if args.operation != "policy":
        check_legacy_inputs(args.config)
    region = config["region"]
    identity = boto3.client("sts", region_name=region).get_caller_identity()
    if identity["Arn"].endswith(":root"):
        raise ValueError("Use o perfil do usuário terraform; bootstrap por root recusado.")
    account = identity["Account"]
    if args.operation == "policy":
        print(json.dumps(server_policy(account, config["project"]), indent=2))
        return
    if not args.repo or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("Informe --repo owner/repository")
    # Faz toda a validação dos valores privados antes de criar infraestrutura.
    run(sys.executable, ROOT / "scripts/cloud.py", "secrets", "--env-file", args.env_file, "--check")
    repo = json.loads(run("gh", "api", f"repos/{args.repo}", capture=True).stdout)
    if str(repo["id"]) != str(config["github_repository_id"]):
        raise ValueError("O ID no arquivo público não corresponde ao repositório GitHub informado.")
    os.environ["AWS_REGION"] = region
    state_bucket = f'{config["project"]}-tfstate-{account}-{region}'
    state_key = f'{config["project"]}/production.tfstate'
    terraform = ("terraform", f"-chdir={TF}")
    if args.operation == "plan":
        RECORD.unlink(missing_ok=True)
        run(sys.executable, ROOT / "infra/bootstrap_state.py", "--bucket", state_bucket, "--key", state_key,
            "--region", region, "--output", TF / "backend.hcl")
        run(*terraform, "init", "-input=false", "-backend-config=backend.hcl")
        if not config.get("ami_id"):
            previous = subprocess.run([*terraform, "output", "-raw", "ami_id"], cwd=ROOT, text=True, capture_output=True)
            config["ami_id"] = previous.stdout.strip() if previous.returncode == 0 and re.fullmatch(r"ami-[0-9a-f]+", previous.stdout.strip()) else boto3.client("ssm", region_name=region).get_parameter(
                    Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
                )["Parameter"]["Value"]
            args.config.write_text(json.dumps(config, indent=2) + "\n")
            print("AMI inicial fixada na configuração pública; versione esse arquivo e o lock do provider.")
        run(*terraform, "fmt", "-check")
        run(*terraform, "validate")
        run(*terraform, "plan", "-input=false", "-lock-timeout=5m", f"-var-file={args.config}", f"-out={PLAN}")
        RECORD.write_text(json.dumps({"account": account, "repo": args.repo, "fingerprint": fingerprint(args.config)}))
        print("Plano salvo. Revise a saída; apply executará este plano, sem gerar outro.")
        return
    if args.operation == "apply":
        if not PLAN.is_file() or not RECORD.is_file():
            raise ValueError("Plano ausente. Execute bootstrap_aws.py plan primeiro.")
        validate_record(json.loads(RECORD.read_text()), account, args.repo, fingerprint(args.config))
        run(*terraform, "apply", "-input=false", PLAN)
    OUTPUTS.write_text(run(*terraform, "output", "-json", capture=True).stdout)
    out = {key: item["value"] for key, item in json.loads(OUTPUTS.read_text()).items()}
    if out["github_oidc_subject"] != f'repository_id:{config["github_repository_id"]}:ref:refs/heads/main' or out["github_role_arn"].split(":")[4] != account or out["region"] != region:
        raise ValueError("Os outputs não correspondem à conta, região ou repositório configurados; nenhum segredo foi enviado.")
    run(sys.executable, ROOT / "scripts/cloud.py", "--outputs", OUTPUTS, "secrets", "--env-file", args.env_file)
    setup = [sys.executable, ROOT / "scripts/configure_github.py", "--repo", args.repo, "--terraform-output", OUTPUTS,
             "--state-bucket", state_bucket, "--state-key", state_key]
    if not args.protect_main:
        setup.append("--mirror-source")
    run(*setup)
    print("Bootstrap configurado. Continue no GitHub Actions: smoke, ETL, promoção e primeira publicação.")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print("Erro: " + str(exc), file=sys.stderr)
        sys.exit(1)
