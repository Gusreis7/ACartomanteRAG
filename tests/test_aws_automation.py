"""Contratos de bootstrap, publicação imutável e execução cloud sem chamadas reais."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def cloud():
    return module("cloud_automation", "scripts/cloud.py")


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with mock_aws():
        yield


def private_file(tmp_path, **changes):
    values = {"QDRANT_URL": "https://cluster.example.org:6333", "QDRANT_READ_API_KEY": "read-key", "QDRANT_WRITE_API_KEY": "write-key", **changes}
    path = tmp_path / "private.env"
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()))
    path.chmod(0o600)
    return path


def test_private_env_is_not_executed_or_interpolated(tmp_path, cloud, monkeypatch):
    path = private_file(tmp_path, EXTRA="$(touch SHOULD_NOT_EXIST)", AWS_SECRET_ACCESS_KEY="must-not-upload")
    result = cloud.read_secret_file(path)
    assert set(result) == {"QDRANT_URL", "QDRANT_READ_API_KEY", "QDRANT_WRITE_API_KEY"}
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()
    path = private_file(tmp_path, QDRANT_WRITE_API_KEY="${SOME_LOCAL_SECRET}")
    monkeypatch.setenv("SOME_LOCAL_SECRET", "do-not-use")
    with pytest.raises(ValueError):
        cloud.read_secret_file(path)


@pytest.mark.parametrize("values", [
    {"QDRANT_URL": "http://local.example.org"},
    {"QDRANT_URL": "https://user:secret@example.org"},
    {"QDRANT_WRITE_API_KEY": ""},
])
def test_invalid_secrets_rejected_without_exposing_values(tmp_path, cloud, values):
    with pytest.raises(ValueError) as error:
        cloud.read_secret_file(private_file(tmp_path, **values))
    assert "user:secret" not in str(error.value)


def test_private_file_permissions_are_checked(tmp_path, cloud):
    path = private_file(tmp_path)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="chmod 600"):
        cloud.read_secret_file(path)


def test_check_secrets_does_not_initialize_aws(tmp_path, cloud, monkeypatch):
    path = private_file(tmp_path)
    client = Mock(side_effect=AssertionError("não deve consultar AWS"))
    monkeypatch.setattr(cloud.boto3, "client", client)
    monkeypatch.setattr("sys.argv", ["cloud.py", "secrets", "--check", "--env-file", str(path)])
    cloud.main()
    client.assert_not_called()


def test_secret_import_preserves_passwords_and_role_separation(tmp_path, cloud, aws):
    ssm = boto3.client("ssm", region_name="us-east-1")
    out = {"app_settings_parameter": "/project/app", "etl_settings_parameter": "/project/etl"}
    values = cloud.read_secret_file(private_file(tmp_path))
    cloud.configure_secrets(ssm, out, values)
    read = lambda name: json.loads(ssm.get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"])
    original = read("/project/app")
    assert original["QDRANT_API_KEY"] == "read-key"
    assert read("/project/etl") == {"QDRANT_URL": values["QDRANT_URL"], "QDRANT_API_KEY": "write-key"}
    cloud.configure_secrets(ssm, out, {**values, "QDRANT_READ_API_KEY": "new-read"})
    current = read("/project/app")
    assert current["REDIS_PASSWORD"] == original["REDIS_PASSWORD"]
    assert current["REDIS_INSPECT_PASSWORD"] == original["REDIS_INSPECT_PASSWORD"]
    assert "AWS_SECRET_ACCESS_KEY" not in current


def test_bad_existing_password_does_not_rotate_or_partially_write(tmp_path, cloud, aws):
    ssm = boto3.client("ssm", region_name="us-east-1")
    ssm.put_parameter(Name="/app", Type="SecureString", Value=json.dumps({"REDIS_PASSWORD": "legacy"}))
    with pytest.raises(ValueError, match="rotação"):
        cloud.configure_secrets(ssm, {"app_settings_parameter": "/app", "etl_settings_parameter": "/etl"}, cloud.read_secret_file(private_file(tmp_path)))
    assert ssm.get_parameter(Name="/app")["Parameter"]["Version"] == 1
    with pytest.raises(ssm.exceptions.ParameterNotFound):
        ssm.get_parameter(Name="/etl")


def make_bundle(directory, cloud):
    directory.mkdir(exist_ok=True)
    for name in cloud.GLUE_FILES:
        (directory / name).write_bytes(name.encode())


def test_bundle_publication_reuses_complete_revision_and_refuses_tampering(tmp_path, cloud, aws):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="automation-ops")
    revision = "a" * 40
    directory = tmp_path / "bundle"
    make_bundle(directory, cloud)
    assert cloud.publish_glue(s3, "automation-ops", revision, directory)
    manifest = cloud.bundle_manifest(s3, "automation-ops", revision)
    (directory / "source.zip").write_bytes(b"repacked-with-different-timestamps")
    assert not cloud.publish_glue(s3, "automation-ops", revision, directory)
    assert cloud.bundle_manifest(s3, "automation-ops", revision) == manifest
    s3.put_object(Bucket="automation-ops", Key=f"glue/{revision}/source.zip", Body=b"tampered")
    with pytest.raises(ValueError, match="alterado"):
        cloud.bundle_manifest(s3, "automation-ops", revision)


def test_partial_bundle_resumes_without_overwrite(tmp_path, cloud, aws):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="automation-ops")
    revision = "b" * 40
    key = f"glue/{revision}/job.py"
    s3.put_object(Bucket="automation-ops", Key=key, Body=b"old-valid-job", Metadata={"revision": revision, "sha256": "c" * 64})
    assert cloud.bundle_manifest(s3, "automation-ops", revision) is None
    make_bundle(tmp_path / "bundle", cloud)
    assert cloud.publish_glue(s3, "automation-ops", revision, tmp_path / "bundle")
    assert s3.get_object(Bucket="automation-ops", Key=key)["Body"].read() == b"old-valid-job"


def test_legacy_bundle_is_not_silently_overwritten(tmp_path, cloud, aws):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="automation-ops")
    revision = "c" * 40
    s3.put_object(Bucket="automation-ops", Key=f"glue/{revision}/job.py", Body=b"legacy")
    make_bundle(tmp_path / "bundle", cloud)
    with pytest.raises(ValueError, match="sem identidade"):
        cloud.publish_glue(s3, "automation-ops", revision, tmp_path / "bundle")
    assert s3.get_object(Bucket="automation-ops", Key=f"glue/{revision}/job.py")["Body"].read() == b"legacy"


def test_glue_arguments_pin_all_assets_without_changing_job(cloud):
    args = cloud.glue_arguments("d" * 40, "ops", "corpus-v1", ",".join(cloud.ETL_STEPS), promote=True)
    assert args["--scriptLocation"] == "s3://ops/glue/" + "d" * 40 + "/job.py"
    assert args["--additional-python-modules"].endswith("/" + "d" * 40 + "/dependencies.gluewheels.zip")
    assert args["--bundle-prefix"] == "glue/" + "d" * 40
    assert args["--steps"].startswith("crawl,bronze,silver")
    assert args["--promote"] == "true"


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("reuse_options", [[], ["--reuse-index-id", ""], ["--reuse-index-id", "corpus-v1"]])
def test_glue_start_omits_empty_optional_value_and_accepts_glue_argv(cloud, monkeypatch, smoke, reuse_options):
    revision = "d" * 40
    clients = {name: Mock() for name in ("ssm", "s3", "glue")}
    clients["glue"].start_job_run.return_value = {"JobRunId": "jr-test"}
    monkeypatch.setattr(cloud.boto3, "client", lambda name, **kwargs: clients[name])
    monkeypatch.setattr(cloud, "outputs", lambda path: {"ops_bucket": "ops", "glue_job": "etl-job"})
    monkeypatch.setattr(cloud, "bundle_manifest", lambda *args: {"revision": revision})
    monkeypatch.setattr("sys.argv", [
        "cloud.py", "glue", "--revision", revision, "--run-id", "smoke-v1",
        *reuse_options, *(["--smoke"] if smoke else []),
    ])
    cloud.main()

    clients["glue"].start_job_run.assert_called_once()
    request = clients["glue"].start_job_run.call_args.kwargs
    assert request["JobName"] == "etl-job"
    arguments = request["Arguments"]
    expected_reuse = reuse_options[-1] if reuse_options else ""
    if expected_reuse:
        assert arguments["--reuse-index-id"] == expected_reuse
    else:
        assert "--reuse-index-id" not in arguments
    assert arguments["--smoke"] == str(smoke).lower()

    # Glue omitiu o valor vazio no argv observado no CloudWatch.
    # Verifica o parser real do job com o mesmo formato de transporte.
    job = module("glue_optional_argument", "infra/glue/job.py")
    parameters = {
        "--assets-bucket": "ops", "--settings-parameter": "/project/etl",
        "--JOB_NAME": "etl-job", "--JOB_RUN_ID": "jr-test", **arguments,
    }
    argv = ["job.py"]
    for name, value in parameters.items():
        argv.append(name)
        if value:
            argv.append(value)
    monkeypatch.setattr("sys.argv", argv)

    class ParsedArguments(Exception):
        pass

    clients["ssm"].get_parameter.side_effect = ParsedArguments
    with pytest.raises(ParsedArguments):
        job.main()
    clients["ssm"].get_parameter.assert_called_once_with(Name="/project/etl", WithDecryption=True)


@pytest.mark.parametrize("kwargs", [
    {"revision": "main"}, {"run_id": "../../bad"}, {"steps": "silver,crawl"},
    {"steps": "crawl,crawl"}, {"steps": ""}, {"smoke": True, "promote": True},
])
def test_invalid_job_inputs_are_rejected_before_start(cloud, kwargs):
    base = dict(revision="e" * 40, bucket="ops", run_id="corpus-v1", steps=",".join(cloud.ETL_STEPS))
    with pytest.raises(ValueError):
        cloud.glue_arguments(**{**base, **kwargs})


def test_ssm_new_host_retry_does_not_hide_permission_errors(cloud, monkeypatch):
    client = Mock()
    invalid = ClientError({"Error": {"Code": "InvalidInstanceId", "Message": "pending"}}, "SendCommand")
    client.send_command.side_effect = [invalid, {"Command": {"CommandId": "ok"}}]
    monkeypatch.setattr(cloud.time, "sleep", lambda _: None)
    assert cloud.send_when_ready(client, InstanceIds=["i-test"])["Command"]["CommandId"] == "ok"
    assert client.send_command.call_count == 2
    client.send_command.side_effect = ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "SendCommand")
    with pytest.raises(ClientError):
        cloud.send_when_ready(client, InstanceIds=["i-test"])


def test_saved_plan_guard_prevents_account_repo_and_source_changes():
    bootstrap = module("bootstrap_automation", "scripts/bootstrap_aws.py")
    record = {"account": "123", "repo": "owner/repo", "fingerprint": "same"}
    bootstrap.validate_record(record, "123", "owner/repo", "same")
    for account, repo, signature in [("456", "owner/repo", "same"), ("123", "other/repo", "same"), ("123", "owner/repo", "changed")]:
        with pytest.raises(ValueError, match="plan novamente"):
            bootstrap.validate_record(record, account, repo, signature)


def test_server_policy_includes_separate_workflow_roles():
    bootstrap = module("bootstrap_policy", "scripts/bootstrap_aws.py")
    policy = bootstrap.server_policy("123456789012", "acartomanterag")
    resources = policy["Statement"][0]["Resource"]
    assert "arn:aws:iam::123456789012:role/acartomanterag-github-infra" in resources
    assert "arn:aws:iam::123456789012:role/acartomanterag-github-etl" in resources
    assert all("123456789012" in value for value in resources)
    assert not any(":user/" in value or ":group/" in value for value in resources)


def test_bootstrap_public_config_refuses_secrets(tmp_path):
    bootstrap = module("bootstrap_config", "scripts/bootstrap_aws.py")
    config = tmp_path / "public.json"
    config.write_text(json.dumps({"QDRANT_API_KEY": "do-not-print"}))
    with pytest.raises(ValueError) as error:
        bootstrap.load_config(config)
    assert "do-not-print" not in str(error.value)


def test_bootstrap_refuses_legacy_implicit_inputs(tmp_path, monkeypatch):
    bootstrap = module("bootstrap_legacy", "scripts/bootstrap_aws.py")
    monkeypatch.setattr(bootstrap, "TF", tmp_path)
    (tmp_path / "terraform.tfvars").write_text('instance_type="different-local-choice"')
    with pytest.raises(ValueError, match="mesma configuração"):
        bootstrap.check_legacy_inputs(tmp_path / "production.tfvars.json")


@pytest.fixture
def bootstrap_fixture(tmp_path, monkeypatch, cloud):
    bootstrap = module("bootstrap_flow", "scripts/bootstrap_aws.py")
    tf = tmp_path / "infra/aws"
    tf.mkdir(parents=True)
    for source in (ROOT / "infra/aws").glob("*.tf"):
        (tf / source.name).write_bytes(source.read_bytes())
    for name in ("cloud-init.sh.tftpl", "manage.py"):
        (tf / name).write_bytes((ROOT / "infra/aws" / name).read_bytes())
    (tf / ".terraform.lock.hcl").write_text("fixture-lock")
    config = tf / "production.tfvars.json"
    config.write_text(json.dumps({"project": "acartomanterag", "region": "us-east-1", "acme_email": "test@example.org", "budget_email": "test@example.org", "github_repository_id": "42", "ami_id": "ami-123abc"}))
    private = private_file(tmp_path)
    for name, value in {"ROOT": tmp_path, "TF": tf, "PLAN": tf / "bootstrap.tfplan", "RECORD": tf / "bootstrap-plan.local.json", "OUTPUTS": tf / "outputs.local.json"}.items():
        monkeypatch.setattr(bootstrap, name, value)
    aws = Mock()
    aws.get_caller_identity.return_value = {"Account": "123456789012", "Arn": "arn:aws:iam::123456789012:user/terraform"}
    monkeypatch.setattr(bootstrap.boto3, "client", lambda *a, **k: aws)
    calls = []
    output = {
        "github_oidc_subject": {"value": "repository_id:42:ref:refs/heads/main"},
        "github_role_arn": {"value": "arn:aws:iam::123456789012:role/acartomanterag-github-deploy"},
        "region": {"value": "us-east-1"},
    }
    def execute(*arguments, capture=False):
        arguments = [str(value) for value in arguments]
        calls.append(arguments)
        if arguments[:2] == ["gh", "api"]:
            return SimpleNamespace(stdout='{"id":42}')
        if arguments[-2:] == ["output", "-json"]:
            return SimpleNamespace(stdout=json.dumps(output))
        if "plan" in arguments:
            bootstrap.PLAN.write_text("saved-plan")
        if "--check" in arguments:
            cloud.read_secret_file(private)
        return SimpleNamespace(stdout="")
    monkeypatch.setattr(bootstrap, "run", execute)
    def invoke(operation):
        monkeypatch.setattr("sys.argv", ["bootstrap_aws.py", operation, "--repo", "owner/repo", "--env-file", str(private), "--config", str(config)])
        bootstrap.main()
    return bootstrap, config, private, calls, output, invoke


def test_bootstrap_plan_then_apply_uses_saved_plan_and_file_path(bootstrap_fixture):
    bootstrap, config, private, calls, output, invoke = bootstrap_fixture
    invoke("plan")
    assert not any("apply" in args for args in calls)
    assert bootstrap.PLAN.read_text() == "saved-plan"
    invoke("apply")
    apply = next(args for args in calls if "apply" in args)
    assert apply[-1] == str(bootstrap.PLAN)
    assert any("--mirror-source" in args for args in calls)
    assert any("--env-file" in args and str(private) in args and "--check" not in args for args in calls)
    assert not any("read-key" in arg or "write-key" in arg for args in calls for arg in args)


def test_bootstrap_source_change_blocks_apply(bootstrap_fixture):
    bootstrap, config, private, calls, output, invoke = bootstrap_fixture
    invoke("plan")
    config.write_text(config.read_text().replace("ami-123abc", "ami-456def"))
    with pytest.raises(ValueError, match="plan novamente"):
        invoke("apply")
    assert not any("apply" in args for args in calls)


def test_configure_refuses_outputs_from_another_account_before_sending_secrets(bootstrap_fixture):
    bootstrap, config, private, calls, output, invoke = bootstrap_fixture
    output["github_role_arn"]["value"] = "arn:aws:iam::999999999999:role/wrong"
    with pytest.raises(ValueError, match="nenhum segredo"):
        invoke("configure")
    assert not any("secrets" in args and "--check" not in args for args in calls)


def test_packager_uses_requested_archive_and_excludes_secrets(tmp_path, monkeypatch):
    package = module("package_automation", "scripts/package_release.py")
    source = tmp_path / "archive"
    for directory in ("src", "config", "infra/glue"):
        (source / directory).mkdir(parents=True)
    (source / "src/example.py").write_text("VALUE = 'selected-revision'\n")
    (source / "config/config.yaml").write_text("example: true\n")
    (source / "config/config.aws.yaml").write_text("extends: config.yaml\n")
    (source / "config/aws-bootstrap.env").write_text("QDRANT_WRITE_API_KEY=never-package\n")
    (source / "infra/glue/job.py").write_text("print('selected-job')\n")
    monkeypatch.setattr("sys.argv", ["package_release.py", "--source-root", str(source), "--sha", "f" * 40, "--only", "glue", "--output", str(tmp_path / "dist")])
    package.main()
    assert not (tmp_path / "dist/bundle.tar.gz").exists()
    with zipfile.ZipFile(tmp_path / "dist/glue/source.zip") as archive:
        assert "config/aws-bootstrap.env" not in archive.namelist()
        assert archive.read("config/source_revision.txt") == b"f" * 40
        assert b"selected-revision" in archive.read("src/example.py")


def test_workflow_contract_orders_infra_before_api_and_separates_etl():
    load = lambda name: yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)
    ci, infrastructure, etl = load("ci-cd.yml"), load("infrastructure.yml"), load("etl.yml")
    assert ci["jobs"]["deploy"]["needs"] == ["validate", "infrastructure"]
    assert "AWS_APP_DEPLOY_ENABLED" in ci["jobs"]["deploy"]["if"]
    assert set(etl["on"]) == {"workflow_dispatch"}
    assert "workflow_call" in infrastructure["on"]
    for job in (ci["jobs"]["deploy"], infrastructure["jobs"]["terraform"]):
        assert job["concurrency"] == {"group": "aws-production", "cancel-in-progress": "false"}
    assert etl["concurrency"] == {"group": "aws-production", "cancel-in-progress": "false"}
    assert all("concurrency" not in job for job in etl["jobs"].values())
    app_runs = "\n".join(step.get("run", "") for step in ci["jobs"]["deploy"]["steps"])
    assert "Dockerfile.wheels" not in app_runs and " cloud.py glue" not in app_runs
    assert "config/deployment.json" in app_runs


@pytest.mark.parametrize("terminal", ["FAILED", "ERROR", "TIMEOUT", "STOPPED", "EXPIRED"])
def test_glue_monitor_propagates_terminal_failures(cloud, terminal):
    client = Mock()
    client.get_job_run.return_value = {"JobRun": {"JobRunState": terminal, "ErrorMessage": "diagnóstico"}}
    with pytest.raises(RuntimeError, match=f"Glue {terminal}: diagnóstico"):
        cloud.wait_job(client, "etl-job", "jr_existing", wait_seconds=1)
    client.start_job_run.assert_not_called()
    client.batch_stop_job_run.assert_not_called()


def test_glue_monitor_window_keeps_running_job_and_reports_heartbeat(cloud, monkeypatch, capsys):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(cloud.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(cloud.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    client = Mock()
    client.get_job_run.return_value = {"JobRun": {"JobRunState": "RUNNING"}}
    assert cloud.wait_job(client, "etl-job", "jr_existing", wait_seconds=305) is False
    assert clock.now == 305
    log = capsys.readouterr().out
    assert log.count("Glue: RUNNING") == 2
    assert "continua ativo" in log
    client.start_job_run.assert_not_called()
    client.batch_stop_job_run.assert_not_called()
    assert all(call.kwargs == {"JobName": "etl-job", "RunId": "jr_existing"} for call in client.get_job_run.call_args_list)


def test_glue_monitor_recognizes_success_at_window_boundary(cloud, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(cloud.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(cloud.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    client = Mock()
    client.get_job_run.side_effect = [
        {"JobRun": {"JobRunState": "RUNNING"}},
        {"JobRun": {"JobRunState": "SUCCEEDED"}},
    ]
    assert cloud.wait_job(client, "etl-job", "jr_existing", wait_seconds=1) is True
    assert clock.now == 1
    client.start_job_run.assert_not_called()


@pytest.mark.parametrize("status, expected_exit", [("SUCCEEDED", None), ("RUNNING", 3)])
def test_wait_glue_cli_only_observes_existing_execution(cloud, monkeypatch, status, expected_exit):
    clients = {name: Mock() for name in ("ssm", "s3", "glue")}
    clients["glue"].get_job_run.return_value = {"JobRun": {"JobRunState": status}}
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(cloud.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(cloud.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    monkeypatch.setattr(cloud.boto3, "client", lambda name, **kwargs: clients[name])
    monkeypatch.setattr(cloud, "outputs", lambda path: {"glue_job": "etl-job"})
    monkeypatch.setattr("sys.argv", ["cloud.py", "wait-glue", "--job-run-id", "jr_existing", "--wait-seconds", "1"])
    if expected_exit:
        with pytest.raises(SystemExit) as error:
            cloud.main()
        assert error.value.code == expected_exit
    else:
        cloud.main()
    clients["glue"].start_job_run.assert_not_called()
    clients["glue"].batch_stop_job_run.assert_not_called()
    assert not clients["s3"].mock_calls
    assert not clients["ssm"].mock_calls


@pytest.mark.parametrize("arguments", [
    ["--job-run-id", "corpus-raw-v1"],
    ["--job-run-id", "jr_existing", "--wait-seconds", "0"],
    ["--job-run-id", "jr_existing", "--wait-seconds", "8101"],
])
def test_wait_glue_rejects_invalid_identity_and_window_before_aws(cloud, monkeypatch, arguments):
    client = Mock(side_effect=AssertionError("não deve consultar AWS"))
    monkeypatch.setattr(cloud.boto3, "client", client)
    monkeypatch.setattr("sys.argv", ["cloud.py", "wait-glue", *arguments])
    with pytest.raises(SystemExit) as error:
        cloud.main()
    assert error.value.code == 2
    client.assert_not_called()


@pytest.mark.parametrize("already_exists", [False, True])
def test_glue_start_exports_only_created_run_id_and_preserves_existing_file(tmp_path, cloud, monkeypatch, already_exists):
    clients = {name: Mock() for name in ("ssm", "s3", "glue")}
    clients["glue"].start_job_run.return_value = {"JobRunId": "jr_created"}
    monkeypatch.setattr(cloud.boto3, "client", lambda name, **kwargs: clients[name])
    monkeypatch.setattr(cloud, "outputs", lambda path: {"ops_bucket": "ops", "glue_job": "etl-job"})
    monkeypatch.setattr(cloud, "bundle_manifest", lambda *args: {"revision": "d" * 40})
    path = tmp_path / "glue-run-id"
    if already_exists:
        path.write_text("jr_previous\n")
    monkeypatch.setattr("sys.argv", ["cloud.py", "glue", "--revision", "d" * 40, "--run-id", "corpus-v1", "--run-id-file", str(path)])
    if already_exists:
        with pytest.raises(SystemExit) as error:
            cloud.main()
        assert error.value.code == 2
        clients["glue"].start_job_run.assert_not_called()
        assert path.read_text() == "jr_previous\n"
    else:
        cloud.main()
        clients["glue"].start_job_run.assert_called_once()
        assert path.read_text() == "jr_created\n"
        assert "Timeout" not in clients["glue"].start_job_run.call_args.kwargs


def test_etl_workflow_keeps_one_glue_run_and_requires_success_after_all_windows():
    workflow = yaml.load((ROOT / ".github/workflows/etl.yml").read_text(), Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    commands = lambda name: "\n".join(step.get("run", "") for step in jobs[name]["steps"])
    assert "--run-id-file" in commands("etl") and "--wait " not in commands("etl")
    assert "--expected-timeout-minutes 600" in commands("etl")
    assert jobs["etl"]["outputs"]["glue_run_id"] == "${{ steps.start.outputs.glue_run_id }}"
    monitor = jobs["monitor"]
    assert monitor["needs"] == "etl"
    assert monitor["strategy"]["max-parallel"] == "1"
    assert len(monitor["strategy"]["matrix"]["window"]) * 135 > 600
    assert "wait-glue" in commands("monitor") and '"$result" -ne 3' in commands("monitor")
    assert jobs["finish"]["needs"] == ["etl", "monitor"]
    final_steps = jobs["finish"]["steps"]
    confirmation = next(i for i, step in enumerate(final_steps) if "Require Glue success" in step.get("name", ""))
    reload = next(i for i, step in enumerate(final_steps) if "Reload application" in step.get("name", ""))
    assert confirmation < reload
    assert "||" not in final_steps[confirmation]["run"]
    for name in ("monitor", "finish"):
        assert jobs[name]["env"]["GLUE_RUN_ID"] == "${{ needs.etl.outputs.glue_run_id }}"
        assert "--run-id-file" not in commands(name)
        assert "promote" not in commands(name)
    for job in jobs.values():
        assert int(job["timeout-minutes"]) < 180
        assert any("configure-aws-credentials@" in step.get("uses", "") for step in job["steps"])


@pytest.mark.parametrize("provisioned_timeout", [120, 600, None])
def test_glue_requires_provisioned_timeout_before_start(cloud, monkeypatch, provisioned_timeout):
    clients = {name: Mock() for name in ("ssm", "s3", "glue")}
    clients["glue"].get_job.return_value = {"Job": {"Timeout": provisioned_timeout}}
    clients["glue"].start_job_run.return_value = {"JobRunId": "jr_created"}
    monkeypatch.setattr(cloud.boto3, "client", lambda name, **kwargs: clients[name])
    monkeypatch.setattr(cloud, "outputs", lambda path: {"ops_bucket": "ops", "glue_job": "etl-job"})
    monkeypatch.setattr(cloud, "bundle_manifest", lambda *args: {"revision": "d" * 40})
    monkeypatch.setattr("sys.argv", ["cloud.py", "glue", "--revision", "d" * 40, "--run-id", "corpus-v1", "--expected-timeout-minutes", "600"])
    if provisioned_timeout == 600:
        cloud.main()
        clients["glue"].start_job_run.assert_called_once()
    else:
        with pytest.raises(ValueError, match="novo plano Terraform"):
            cloud.main()
        clients["glue"].start_job_run.assert_not_called()
    clients["glue"].update_job.assert_not_called()
