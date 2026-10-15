"""Configura GitHub por gh CLI, sem mudar a visibilidade do repositório."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess


def gh(endpoint, *, method="GET", payload=None):
    args = ["gh", "api", endpoint, "--method", method]
    if payload is not None:
        args += ["--input", "-"]
    result = subprocess.run(args, input=json.dumps(payload) if payload is not None else None, text=True, capture_output=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="owner/repository")
    p.add_argument("--terraform-output", type=Path)
    p.add_argument("--inspect", action="store_true", help="Lê o ID sem alterar configurações")
    args = p.parse_args()
    repo = gh(f"repos/{args.repo}")
    repository_id = str(repo["id"])
    subject = f"repository_id:{repository_id}:ref:refs/heads/main"
    print(f"github_repository_id = \"{repository_id}\"")
    print(f"OIDC subject: {subject}")
    if args.inspect:
        return
    gh(f"repos/{args.repo}", method="PATCH", payload={"allow_squash_merge": True, "allow_merge_commit": False})
    ref = gh(f"repos/{args.repo}/git/ref/heads/main")
    refs = gh(f"repos/{args.repo}/git/matching-refs/heads/develop")
    if not any(item["ref"] == "refs/heads/develop" for item in refs):
        gh(f"repos/{args.repo}/git/refs", method="POST", payload={"ref": "refs/heads/develop", "sha": ref["object"]["sha"]})
    gh(f"repos/{args.repo}/actions/oidc/customization/sub", method="PUT", payload={
        "use_default": False, "include_claim_keys": ["repository_id", "ref"],
    })
    gh(f"repos/{args.repo}/actions/permissions/workflow", method="PUT", payload={
        "default_workflow_permissions": "read", "can_approve_pull_request_reviews": False,
    })
    gh(f"repos/{args.repo}/branches/main/protection", method="PUT", payload={
        "required_status_checks": {"strict": True, "contexts": ["validate"]},
        "enforce_admins": True,
        "required_pull_request_reviews": {"dismiss_stale_reviews": True, "required_approving_review_count": 0},
        "restrictions": None, "required_linear_history": True,
        "allow_force_pushes": False, "allow_deletions": False,
        "required_conversation_resolution": True,
    })
    if args.terraform_output:
        outputs = {key: item["value"] for key, item in json.loads(args.terraform_output.read_text()).items()}
        if outputs["github_oidc_subject"] != subject:
            raise ValueError("O Terraform pertence a outro repositório; nenhuma variável de deploy foi configurada.")
        variables = {
            "AWS_REGION": outputs["region"], "AWS_DEPLOY_ROLE_ARN": outputs["github_role_arn"],
            "ECR_REPOSITORY": outputs["ecr_repository"], "OPS_BUCKET": outputs["ops_bucket"],
            "EC2_INSTANCE_ID": outputs["instance_id"], "SSM_DEPLOY_DOCUMENT": outputs["ssm_deploy_document"],
        }
        for name, value in variables.items():
            subprocess.run(["gh", "variable", "set", name, "--repo", args.repo, "--body", value], check=True)
    print("GitHub configurado: PR + check validate na main; deploy por OIDC restrito à main.")


if __name__ == "__main__":
    main()
