"""Configura GitHub por gh CLI, sem mudar a visibilidade do repositório."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess


def read_terraform_config(path, repository_id):
    payload = path.read_text(encoding="utf-8")
    try:
        config = json.loads(payload)
    except json.JSONDecodeError:
        raise ValueError("Tfvars de produção inválido; conteúdo omitido.") from None
    if not isinstance(config, dict):
        raise ValueError("Tfvars de produção deve conter um objeto JSON.")
    if str(config.get("github_repository_id", "")) != repository_id:
        raise ValueError(
            "Os tfvars pertencem a outro repositório; nenhuma configuração foi alterada."
        )
    return payload


def gh(endpoint, *, method="GET", payload=None):
    args = ["gh", "api", endpoint, "--method", method]
    if payload is not None:
        args += ["--input", "-"]
    result = subprocess.run(
        args,
        input=json.dumps(payload) if payload is not None else None,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="owner/repository")
    p.add_argument("--terraform-output", type=Path)
    p.add_argument(
        "--terraform-config",
        type=Path,
        help="Tfvars privado enviado ao Secret TF_PRODUCTION_TFVARS_JSON",
    )
    p.add_argument("--state-bucket")
    p.add_argument("--state-key", default="acartomanterag/production.tfstate")
    p.add_argument(
        "--mirror-source",
        action="store_true",
        help="Preserva proteções existentes; não exige PR no GitHub para o push espelhado",
    )
    p.add_argument(
        "--inspect", action="store_true", help="Lê o ID sem alterar configurações"
    )
    args = p.parse_args()
    repo = gh(f"repos/{args.repo}")
    repository_id = str(repo["id"])
    subject = f"repository_id:{repository_id}:ref:refs/heads/main"
    print(f'github_repository_id = "{repository_id}"')
    print(f"OIDC subject: {subject}")
    if args.inspect:
        return
    terraform_config = (
        read_terraform_config(args.terraform_config, repository_id)
        if args.terraform_config
        else None
    )
    outputs = None
    if args.terraform_output:
        outputs = {
            key: item["value"]
            for key, item in json.loads(args.terraform_output.read_text()).items()
        }
        if outputs["github_oidc_subject"] != subject:
            raise ValueError(
                "O Terraform pertence a outro repositório; nenhuma configuração foi alterada."
            )
    gh(
        f"repos/{args.repo}",
        method="PATCH",
        payload={"allow_squash_merge": True, "allow_merge_commit": False},
    )
    if not args.mirror_source:
        ref = gh(f"repos/{args.repo}/git/ref/heads/main")
        refs = gh(f"repos/{args.repo}/git/matching-refs/heads/develop")
        if not any(item["ref"] == "refs/heads/develop" for item in refs):
            gh(
                f"repos/{args.repo}/git/refs",
                method="POST",
                payload={"ref": "refs/heads/develop", "sha": ref["object"]["sha"]},
            )
    gh(
        f"repos/{args.repo}/actions/oidc/customization/sub",
        method="PUT",
        payload={
            "use_default": False,
            "include_claim_keys": ["repository_id", "ref"],
        },
    )
    gh(
        f"repos/{args.repo}/actions/permissions/workflow",
        method="PUT",
        payload={
            "default_workflow_permissions": "read",
            "can_approve_pull_request_reviews": False,
        },
    )
    if not args.mirror_source:
        gh(
            f"repos/{args.repo}/branches/main/protection",
            method="PUT",
            payload={
                "required_status_checks": {"strict": True, "contexts": ["validate"]},
                "enforce_admins": True,
                "required_pull_request_reviews": {
                    "dismiss_stale_reviews": True,
                    "required_approving_review_count": 0,
                },
                "restrictions": None,
                "required_linear_history": True,
                "allow_force_pushes": False,
                "allow_deletions": False,
                "required_conversation_resolution": True,
            },
        )
    if outputs:
        variables = {
            "AWS_REGION": outputs["region"],
            "AWS_DEPLOY_ROLE_ARN": outputs["github_role_arn"],
            "ECR_REPOSITORY": outputs["ecr_repository"],
            "OPS_BUCKET": outputs["ops_bucket"],
            "EC2_INSTANCE_ID": outputs["instance_id"],
            "SSM_DEPLOY_DOCUMENT": outputs["ssm_deploy_document"],
        }
        for output, variable in (
            ("github_infra_role_arn", "AWS_INFRA_ROLE_ARN"),
            ("github_etl_role_arn", "AWS_ETL_ROLE_ARN"),
        ):
            if output in outputs:
                variables[variable] = outputs[output]
        if args.state_bucket:
            variables.update(
                TF_STATE_BUCKET=args.state_bucket, TF_STATE_KEY=args.state_key
            )
        for name, value in variables.items():
            subprocess.run(
                ["gh", "variable", "set", name, "--repo", args.repo, "--body", value],
                check=True,
            )
        page = 1
        while True:
            existing = gh(
                f"repos/{args.repo}/actions/variables?per_page=100&page={page}"
            )["variables"]
            if any(item["name"] == "AWS_APP_DEPLOY_ENABLED" for item in existing):
                break
            if len(existing) < 100:
                subprocess.run(
                    [
                        "gh",
                        "variable",
                        "set",
                        "AWS_APP_DEPLOY_ENABLED",
                        "--repo",
                        args.repo,
                        "--body",
                        "false",
                    ],
                    check=True,
                )
                break
            page += 1
    if terraform_config is not None:
        subprocess.run(
            ["gh", "secret", "set", "TF_PRODUCTION_TFVARS_JSON", "--repo", args.repo],
            input=terraform_config,
            text=True,
            check=True,
            capture_output=True,
        )
        print("Configuração Terraform enviada ao Secret do GitHub; conteúdo omitido.")
    print(
        "GitHub configurado por OIDC na main. Deploy da API permanece desativado no primeiro bootstrap, até os dados estarem publicados."
    )


if __name__ == "__main__":
    main()
