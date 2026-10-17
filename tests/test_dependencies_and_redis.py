"""Migração do manifesto e limites de rede/administração, sem Docker/AWS reais."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
from unittest.mock import Mock

import hcl2
from packaging.requirements import Requirement
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def dependencies():
    return module("dependencies_under_test", "scripts/dependencies.py")


def active_pins(lines, python):
    result = []
    seen = set()
    for line in lines:
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate(
            {"python_version": python, "python_full_version": python + ".0"}
        ):
            continue
        assert not requirement.extras and str(requirement.specifier).startswith("==")
        name = requirement.name.lower()
        assert name not in seen
        seen.add(name)
        result.append(name + str(requirement.specifier))
    return sorted(result)


@pytest.mark.parametrize(
    "profile,python,count,digest",
    [
        (
            "runtime",
            "3.12",
            101,
            "41297f87f7c30e5af4bd6279d16449e1a0956336e32ff4a8a02928a73b062984",
        ),
        (
            "glue",
            "3.11",
            105,
            "d786a709fd32aea4f9feeed2341f119df598f34c1ebdb919078d7086d63b1342",
        ),
    ],
)
def test_exports_preserve_entire_legacy_lock_for_each_production_python(
    dependencies, profile, python, count, digest
):
    groups, _ = dependencies.load()
    lines = active_pins(dependencies.resolve(groups, profile), python)
    assert len(lines) == count
    assert hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest() == digest


def test_light_profiles_do_not_install_api_models_or_analysis_stack(dependencies):
    groups, _ = dependencies.load()
    for name in ("ops", "ci", "crawler"):
        packages = {
            Requirement(line).name.lower()
            for line in dependencies.resolve(groups, name)
        }
        assert not packages & {"torch", "sentence-transformers", "spacy", "plotly"}


def test_install_passes_selected_profile_and_constraints_without_shell_or_persistent_lists(
    dependencies, monkeypatch
):
    captured = {}

    def pip(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        path = Path(command[command.index("--requirement") + 1])
        captured["temporary"] = path
        captured["requirements"] = path.read_text()
        captured["constraints"] = Path(
            command[command.index("--constraint") + 1]
        ).read_text()

    monkeypatch.setattr(dependencies.subprocess, "run", pip)
    monkeypatch.setattr("sys.argv", ["dependencies.py", "install", "ops"])
    dependencies.main()
    assert captured["command"][:4] == [
        dependencies.sys.executable,
        "-m",
        "pip",
        "install",
    ]
    assert captured["kwargs"] == {"check": True}
    assert (
        "boto3" in captured["requirements"] and "torch" not in captured["requirements"]
    )
    assert (
        "torch==" in captured["constraints"]
        and "python_version" in captured["constraints"]
    )
    assert "--extra-index-url" not in captured["command"]
    assert not captured["temporary"].exists()


def test_glue_export_is_complete_and_keeps_cpu_wheel_source(
    dependencies, tmp_path, monkeypatch
):
    target = tmp_path / "export with spaces.txt"
    monkeypatch.setattr(
        "sys.argv", ["dependencies.py", "export", "glue", "--output", str(target)]
    )
    dependencies.main()
    content = target.read_text()
    assert content.startswith(
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
    )
    assert "torch==2.14.1+cpu" in content
    assert "-r " not in content and "-c " not in content
    assert len(active_pins(content.splitlines()[1:], "3.11")) == 105


def test_manifest_check_does_not_install_and_detects_cycles_and_bad_includes(
    dependencies, monkeypatch
):
    pip = Mock()
    monkeypatch.setattr(dependencies.subprocess, "run", pip)
    monkeypatch.setattr("sys.argv", ["dependencies.py", "check"])
    dependencies.main()
    pip.assert_not_called()
    with pytest.raises(ValueError, match="Ciclo"):
        dependencies.resolve(
            {"one": [{"include-group": "two"}], "two": [{"include-group": "one"}]},
            "one",
        )
    with pytest.raises(ValueError, match="inválida"):
        dependencies.resolve({"one": ["-r private.env"]}, "one")
    assert dependencies.resolve({"my-group": ["boto3"]}, "MY_group") == ["boto3"]


def test_all_installer_consumers_use_central_manifest_and_deleted_files_are_not_referenced():
    deleted = [
        "requirements.txt",
        "requirements-runtime.txt",
        "requirements-runtime.lock",
        "requirements-glue.in",
        "requirements-glue.lock",
        "requirements-ci.txt",
        "requirements-crawler.txt",
    ]
    assert all(not (ROOT / name).exists() for name in deleted)
    paths = [
        ROOT / "Dockerfile",
        ROOT / "infra/glue/Dockerfile.wheels",
        *ROOT.glob(".github/workflows/*.yml"),
        ROOT / ".gitea/workflows/deploy.yml",
    ]
    for path in paths:
        assert not any(
            re.search(
                r"(?<![\w./-])" + re.escape(name) + r"(?![\w.-])", path.read_text()
            )
            for name in deleted
        ), path
    dockerignore = (ROOT / ".dockerignore").read_text().splitlines()
    assert (
        "!pyproject.toml" in dockerignore and "!scripts/dependencies.py" in dockerignore
    )


def test_redis_has_only_application_network_and_loopback_administration():
    compose = yaml.load(
        (ROOT / "docker-compose.aws.yml").read_text(), Loader=yaml.BaseLoader
    )
    services = compose["services"]
    assert services["redis"]["ports"] == ["127.0.0.1:6379:6379"]
    assert compose["networks"]["redis_private"]["internal"] == "true"
    members = {
        name
        for name, service in services.items()
        if "redis_private" in service.get("networks", [])
    }
    assert members == {"api", "redis"}
    assert "--aclfile /data/users.acl" in services["redis"]["environment"]["REDIS_ARGS"]
    with (ROOT / "infra/aws/main.tf").open() as source:
        terraform = hcl2.load(source)

    def normalized(value):
        if isinstance(value, dict):
            return {key.strip('"'): normalized(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalized(item) for item in value]
        if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
            return json.loads(value)
        return value

    terraform = normalized(terraform)
    group = next(
        resource["aws_security_group"]["app"]
        for resource in terraform["resource"]
        if "aws_security_group" in resource
    )
    assert all(
        rule["from_port"] == rule["to_port"] == 443 and rule["protocol"] == "tcp"
        for rule in group["ingress"]
    )


def test_notebook_redis_access_uses_authenticated_ssm_fixed_local_port(monkeypatch):
    cloud = module("cloud_tunnel_isolation", "scripts/cloud.py")
    monkeypatch.setattr(cloud.boto3, "client", Mock())
    monkeypatch.setattr(
        cloud, "outputs", lambda _: {"instance_id": "i-0123456789abcdef0"}
    )
    execute = Mock()
    monkeypatch.setattr(cloud.subprocess, "run", execute)
    monkeypatch.setattr("sys.argv", ["cloud.py", "redis-tunnel"])
    cloud.main()
    command = execute.call_args.args[0]
    assert command[:3] == ["aws", "ssm", "start-session"]
    assert (
        command[command.index("--document-name") + 1]
        == "AWS-StartPortForwardingSession"
    )
    assert json.loads(command[command.index("--parameters") + 1]) == {
        "portNumber": ["6379"],
        "localPortNumber": ["16379"],
    }
