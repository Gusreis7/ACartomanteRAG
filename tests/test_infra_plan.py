from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import sys

import hcl2
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "infra_plan", ROOT / "scripts/check_infra_plan.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def plan(address="aws_glue_job.etl", actions=None, **change):
    return {
        "format_version": "1.2",
        "complete": True,
        "resource_changes": [
            {
                "mode": "managed",
                "address": address,
                "change": {"actions": actions or ["update"], **change},
            }
        ],
    }


@pytest.mark.parametrize("address", sorted(module.CI_UPDATES))
def test_existing_operational_resources_can_update(address):
    assert module.administrative_changes(plan(address)) == []


@pytest.mark.parametrize(
    "address",
    [
        "aws_iam_role.github_infra",
        "aws_iam_role_policy.github_infra_operations",
        "aws_iam_role_policy.app",
        "aws_iam_openid_connect_provider.github[0]",
        "aws_security_group.app",
        'aws_s3_bucket_public_access_block.data["raw"]',
        'aws_s3_bucket_policy.tls["ops"]',
        "aws_budgets_budget.monthly",
        "aws_instance.other",
        "module.external.aws_glue_job.etl",
    ],
)
def test_foundation_and_iam_require_local_administrator(address):
    assert module.administrative_changes(plan(address)) == [address]


@pytest.mark.parametrize(
    "actions",
    [["create"], ["delete"], ["delete", "create"], ["create", "delete"], ["forget"]],
)
def test_lifecycle_changes_are_not_sent_to_ci_apply(actions):
    assert module.administrative_changes(plan(actions=actions)) == ["aws_glue_job.etl"]


def test_complete_plan_is_checked_before_any_apply():
    payload = plan()
    payload["resource_changes"].extend(
        plan("aws_iam_role_policy.github_infra_iam")["resource_changes"]
    )
    assert module.administrative_changes(payload) == [
        "aws_iam_role_policy.github_infra_iam"
    ]


def test_noop_and_data_reads_preserve_regular_deploys():
    payload = plan("aws_iam_role.github_infra", ["no-op"])
    data = plan("data.aws_caller_identity.current", ["read"])["resource_changes"][0]
    data["mode"] = "data"
    payload["resource_changes"].append(data)
    assert module.administrative_changes(payload) == []


@pytest.mark.parametrize("operation", ["import", "move"])
def test_state_rebinding_requires_local_administrator(operation):
    payload = plan(actions=["no-op"])
    entry = payload["resource_changes"][0]
    if operation == "import":
        entry["change"]["importing"] = {"id": "job-from-another-project"}
    else:
        entry["previous_address"] = "aws_glue_job.other"
    assert module.administrative_changes(payload) == ["aws_glue_job.etl"]


@pytest.mark.parametrize(
    "payload",
    [None, {}, {"format_version": "2.0"}, {"format_version": "1.2", "complete": False}],
)
def test_invalid_or_deferred_plan_is_rejected(payload):
    with pytest.raises(ValueError):
        module.administrative_changes(payload)


def test_cli_does_not_print_private_plan_values(tmp_path):
    payload = plan("aws_iam_role_policy.app")
    payload["private_value"] = "CANARY_PRIVATE_PLAN_VALUE"
    path = tmp_path / "private.json"
    path.write_text(json.dumps(payload))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_infra_plan.py"), str(path)],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "aws_iam_role_policy.app" in result.stderr
    assert "CANARY_PRIVATE_PLAN_VALUE" not in result.stdout + result.stderr


def test_workflow_checks_saved_plan_and_removes_its_json():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/infrastructure.yml").read_text()
    )
    steps = workflow["jobs"]["terraform"]["steps"]
    names = [item.get("name") for item in steps]
    scope = names.index("Check scope before CI apply")
    assert names.index("Plan") < scope < names.index("Apply")
    assert steps[scope]["if"] == "inputs.operation == 'apply'"
    assert "umask 077" in steps[scope]["run"]
    assert 'show -json "$RUNNER_TEMP/deployment.tfplan"' in steps[scope]["run"]
    assert "check_infra_plan.py" in steps[scope]["run"]
    cleanup = steps[-1]
    assert cleanup["if"] == "always()"
    assert '"$RUNNER_TEMP/deployment.plan.json"' in cleanup["run"]


def test_permissions_contract_plans_without_creating_mock_state():
    with (ROOT / "infra/aws/tests/infra_permissions.tftest.hcl").open() as source:
        config = hcl2.load(source)

    def normalize(value):
        # python-hcl2 7 and 8 serialize labels and expressions differently.
        if isinstance(value, dict):
            return {key.strip('"'): normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str):
            if value.startswith('"') and value.endswith('"'):
                return json.loads(value)
            if value == "${plan}":
                return "plan"
        return value

    config = normalize(config)
    assert not config.get("provider"), "O contrato não deve usar provider real."
    mock = config["mock_provider"][0]["aws"]
    assert mock["override_during"] == "plan"
    run = config["run"][0]["project_permissions"]
    assert run["command"] == "plan", "Apply simulado conflita com prevent_destroy."
    assert not run.get("providers"), "O run deve conservar o provider simulado."
    for override in run.get("override_resource", []):
        assert (
            override["override_during"] == "plan"
        ), "Terraform 1.11 precisa do timing explícito também nas roles."

    # Follow policy locals so indirect references (such as github_oidc_arn)
    # are checked too. Terraform 1.11 only fills registered mock types in plan.
    definitions = {}
    for path in (ROOT / "infra/aws").glob("*.tf"):
        with path.open() as source:
            for block in hcl2.load(source).get("locals", []):
                definitions.update(
                    {name.strip('"'): value for name, value in block.items()}
                )
    pending = ["github_infra_read_policy", "github_infra_operations_policy"]
    visited, required = set(), set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        expression = json.dumps(definitions[name])
        required.update(
            re.findall(r"(?<!data\.)\baws_[a-z0-9_]+\.[a-zA-Z0-9_]+", expression)
        )
        pending.extend(re.findall(r"\blocal\.([a-zA-Z0-9_]+)", expression))
    mocked_types = {
        kind
        for item in mock.get("mock_resource", [])
        for kind, settings in item.items()
        if isinstance(settings, dict)
        and settings.get("override_during", mock["override_during"]) == "plan"
    }
    overridden = {
        item["target"].removeprefix("${").removesuffix("}")
        for item in run.get("override_resource", [])
        if item.get("override_during") == "plan"
    }
    missing = {
        address
        for address in required
        if address.split(".")[0] not in mocked_types and address not in overridden
    }
    assert (
        not missing
    ), f"Referências das políticas sem mock de plano: {sorted(missing)}"
