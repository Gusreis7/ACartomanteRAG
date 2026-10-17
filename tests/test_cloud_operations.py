import importlib.util
import io
import json
from pathlib import Path
import tarfile
from unittest.mock import Mock, patch

from moto import mock_aws
import boto3
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.mark.parametrize(
    "member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE]
)
def test_host_deploy_refuses_non_file_members(tmp_path, member_type):
    manager = module("host_manager", "infra/aws/manage.py")
    path = tmp_path / "malicious.tar.gz"
    names = [
        "docker-compose.aws.yml",
        "infra/aws/Caddyfile",
        "infra/aws/manage.py",
        "release.json",
    ]
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            info = tarfile.TarInfo(name)
            if name == names[0]:
                info.type = member_type
                info.linkname = "/etc/passwd"
                archive.addfile(info)
            else:
                data = b"{}"
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="Tipo"):
        manager.validate_archive(path, tmp_path / "extract")
    assert not (tmp_path / "extract").exists()


def test_host_deploy_requires_expected_files_before_extraction(tmp_path):
    manager = module("host_manager", "infra/aws/manage.py")
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("../../outside")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="esperados"):
        manager.validate_archive(path, tmp_path / "extract")


@mock_aws
def test_host_env_uses_ssm_and_separate_readonly_inspector(tmp_path, monkeypatch):
    manager = module("host_manager", "infra/aws/manage.py")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    manager.ROOT = tmp_path / "data"
    manager.CONFIG = tmp_path / "config"
    manager.ENV = manager.CONFIG / "runtime.env"
    manager.CONFIG.mkdir()
    (manager.CONFIG / "host.json").write_text(
        json.dumps(
            {
                "AWS_REGION": "us-east-1",
                "SETTINGS_PARAMETER": "/app",
                "ECR_REPOSITORY": "example.org/app",
                "OPS_BUCKET": "ops",
            }
        )
    )
    ssm = boto3.client("ssm", region_name="us-east-1")
    ssm.put_parameter(
        Name="/app",
        Type="SecureString",
        Value=json.dumps(
            {
                "QDRANT_URL": "https://cluster.example.org",
                "QDRANT_API_KEY": "testing",
                "REDIS_PASSWORD": "a" * 64,
                "REDIS_INSPECT_PASSWORD": "b" * 64,
            }
        ),
    )
    manager.render_environment("example.org/app@sha256:" + "a" * 64)
    assert manager.ENV.stat().st_mode & 0o777 == 0o600
    acl = (manager.ROOT / "data/redis/users.acl").read_text()
    assert "user inspector on #" in acl
    assert "a" * 64 not in acl
    assert "-@dangerous" in acl
    assert "AWS_ACCESS_KEY_ID" not in manager.ENV.read_text()


def test_github_script_refuses_wrong_terraform_repo_for_variables(
    tmp_path, monkeypatch
):
    setup = module("github_setup", "scripts/configure_github.py")
    # get metadata, branch main and matching develop ref.
    fake = Mock(
        side_effect=[
            {"id": 42},
            None,
            {"object": {"sha": "a" * 40}},
            [{"ref": "refs/heads/develop"}],
            None,
            None,
            None,
        ]
    )
    monkeypatch.setattr(setup, "gh", fake)
    data = tmp_path / "outputs.json"
    data.write_text(json.dumps({"github_oidc_subject": {"value": "wrong"}}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "configure_github.py",
            "--repo",
            "owner/repo",
            "--terraform-output",
            str(data),
        ],
    )
    with patch("subprocess.run") as subprocess_call:
        with pytest.raises(ValueError, match="outro repositório"):
            setup.main()
        subprocess_call.assert_not_called()


def test_github_script_sends_private_tfvars_through_stdin(
    tmp_path, monkeypatch, capsys
):
    setup = module("github_private_tfvars", "scripts/configure_github.py")
    config = tmp_path / "private.json"
    payload = json.dumps(
        {
            "github_repository_id": "42",
            "acme_email": "private-marker@example.org",
            "ami_id": "ami-123abc",
        }
    )
    config.write_text(payload)
    github = Mock(side_effect=[{"id": 42}, None, None, None])
    monkeypatch.setattr(setup, "gh", github)
    monkeypatch.setattr(
        "sys.argv",
        [
            "configure_github.py",
            "--repo",
            "owner/repo",
            "--mirror-source",
            "--terraform-config",
            str(config),
        ],
    )
    with patch.object(setup.subprocess, "run") as execute:
        setup.main()
    execute.assert_called_once_with(
        ["gh", "secret", "set", "TF_PRODUCTION_TFVARS_JSON", "--repo", "owner/repo"],
        input=payload,
        text=True,
        check=True,
        capture_output=True,
    )
    assert "private-marker" not in capsys.readouterr().out
    assert not any(
        "/git/" in call.args[0] or "/protection" in call.args[0]
        for call in github.call_args_list
    )


@pytest.mark.parametrize(
    "payload,error",
    [
        ('{"private-marker":', "inválido"),
        ('["private-marker"]', "objeto JSON"),
        (
            '{"github_repository_id":"99","acme_email":"private-marker"}',
            "outro repositório",
        ),
    ],
)
def test_github_script_refuses_bad_private_tfvars_before_mutations(
    tmp_path, monkeypatch, payload, error
):
    setup = module("github_private_bad_tfvars", "scripts/configure_github.py")
    config = tmp_path / "private.json"
    config.write_text(payload)
    inspect = Mock(return_value={"id": 42})
    monkeypatch.setattr(setup, "gh", inspect)
    monkeypatch.setattr(
        "sys.argv",
        [
            "configure_github.py",
            "--repo",
            "owner/repo",
            "--terraform-config",
            str(config),
        ],
    )
    with patch.object(setup.subprocess, "run") as execute:
        with pytest.raises(ValueError, match=error) as caught:
            setup.main()
    assert "private-marker" not in str(caught.value)
    inspect.assert_called_once_with("repos/owner/repo")
    execute.assert_not_called()
