"""Recusa alterações administrativas no plano da CI; não substitui a política IAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Endereços existentes. Criação, exclusão, import e substituição usam o perfil local.
CI_UPDATES = frozenset(
    {
        "aws_instance.app",
        "aws_ebs_volume.data",
        "aws_glue_job.etl",
        "aws_ssm_document.deploy",
        "aws_ssm_document.reload_data",
        "aws_s3_object.buckets",
        "aws_s3_object.deployment_context",
        "aws_ecr_repository.api",
        "aws_ecr_lifecycle_policy.api",
        "aws_route53_record.app",
        "aws_cloudwatch_log_group.operations",
    }
)


def administrative_changes(plan: dict) -> list[str]:
    if not isinstance(plan, dict) or not isinstance(plan.get("format_version"), str):
        raise ValueError("Use o JSON produzido por terraform show -json.")
    if plan["format_version"].split(".", 1)[0] != "1":
        raise ValueError("Formato de plano Terraform não suportado.")
    if plan.get("errored") or plan.get("complete") is False:
        raise ValueError("Plano incompleto ou com erro; gere um plano completo.")
    changes = plan.get("resource_changes", [])
    if not isinstance(changes, list):
        raise ValueError("Lista de alterações inválida no plano.")
    denied = []
    for resource in changes:
        if not isinstance(resource, dict) or not isinstance(
            resource.get("change"), dict
        ):
            raise ValueError("Alteração inválida no plano.")
        change = resource["change"]
        actions = change.get("actions")
        if (
            not isinstance(actions, list)
            or not actions
            or not all(isinstance(action, str) for action in actions)
        ):
            raise ValueError("Ações inválidas no plano.")
        address = resource.get("address")
        if not isinstance(address, str) or not address:
            raise ValueError("Endereço inválido no plano.")
        if change.get("importing") or resource.get("previous_address"):
            denied.append(address)
        elif resource.get("mode") == "data" and actions == ["read"]:
            continue
        elif actions == ["no-op"]:
            continue
        elif (
            resource.get("mode") == "managed"
            and address in CI_UPDATES
            and actions == ["update"]
        ):
            continue
        else:
            denied.append(address)
    return sorted(set(denied))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plan", type=Path, help="Arquivo privado de terraform show -json"
    )
    args = parser.parse_args()
    try:
        denied = administrative_changes(json.loads(args.plan.read_text()))
    except (OSError, ValueError, TypeError) as exc:
        print(
            f"Erro: plano recusado ({type(exc).__name__}); confira o arquivo local.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    if denied:
        print(
            "Plano exige o perfil local terraform: IAM, fundação, criação, exclusão, "
            "importação ou substituição fora do escopo da CI.",
            file=sys.stderr,
        )
        for address in denied:
            print(f"  {address}", file=sys.stderr)
        print(
            "Aplique localmente o plano revisado e execute novamente a pipeline. "
            "Nenhum apply foi iniciado por esta checagem.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("Plano compatível com as atualizações permitidas à CI.")


if __name__ == "__main__":
    main()
