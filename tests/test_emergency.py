"""Emergência: parar computação, conservar dados e bloquear novas operações."""
import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = {"ops_bucket": "project-ops", "instance_id": "i-0123456789abcdef0", "glue_job": "project-etl"}


@pytest.fixture
def cloud():
    spec = importlib.util.spec_from_file_location("cloud_emergency", ROOT / "scripts/cloud.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instance(state):
    return {"Reservations": [{"Instances": [{"State": {"Name": state}}]}]}


def control(active):
    s3 = Mock()
    s3.list_objects_v2.return_value = {"Contents": [{"Key": "config/emergency.json"}]}
    s3.get_object.side_effect = lambda **_: {"Body": io.BytesIO(json.dumps({"active": active}).encode())}
    return s3


def test_unset_flag_allows_first_deploy_but_malformed_or_unreadable_flag_blocks(cloud):
    s3 = Mock()
    s3.list_objects_v2.return_value = {}
    cloud.require_operations_enabled(s3, "ops")
    s3.get_object.assert_not_called()
    s3 = control(True)
    with pytest.raises(RuntimeError, match="emergência ativa"):
        cloud.require_operations_enabled(s3, "ops")
    s3.get_object.side_effect = lambda **_: {"Body": io.BytesIO(b'{"active":"false"}')}
    with pytest.raises(ValueError):
        cloud.require_operations_enabled(s3, "ops")
    s3.list_objects_v2.side_effect = PermissionError("AccessDenied")
    with pytest.raises(PermissionError):
        cloud.require_operations_enabled(s3, "ops")


def test_stop_sets_latch_before_stopping_both_services_and_keeps_all_data(cloud):
    events = []
    s3, ec2, glue = Mock(), Mock(), Mock()
    s3.put_object.side_effect = lambda **kw: events.append(("flag", json.loads(kw["Body"])["active"]))
    ec2.stop_instances.side_effect = lambda **_: events.append(("ec2", "stop"))
    ec2.describe_instances.return_value = instance("stopped")
    glue.get_paginator.return_value.paginate.side_effect = [
        [{"JobRuns": [{"Id": "jr_current", "JobRunState": "RUNNING"}, {"Id": "jr_old", "JobRunState": "SUCCEEDED"}]}],
        [{"JobRuns": []}],
    ]
    glue.batch_stop_job_run.return_value = {"Errors": []}
    cloud.emergency_operation(s3, ec2, glue, OUT, "stop")
    assert events == [("flag", True), ("ec2", "stop")]
    glue.batch_stop_job_run.assert_called_once_with(JobName="project-etl", JobRunIds=["jr_current"])
    s3.delete_object.assert_not_called()
    ec2.terminate_instances.assert_not_called()
    ec2.delete_volume.assert_not_called()
    assert s3.put_object.call_count == 1


def test_partial_glue_failure_still_stops_ec2_and_keeps_latch(cloud):
    s3, ec2, glue = Mock(), Mock(), Mock()
    glue.get_paginator.return_value.paginate.side_effect = RuntimeError("Glue unavailable")
    with pytest.raises(RuntimeError, match="Parada parcial; trava mantida"):
        cloud.emergency_operation(s3, ec2, glue, OUT, "stop")
    ec2.stop_instances.assert_called_once()
    assert json.loads(s3.put_object.call_args.kwargs["Body"])["active"] is True


def test_stop_also_catches_job_started_during_shutdown(cloud, monkeypatch):
    s3, ec2, glue = Mock(), Mock(), Mock()
    ec2.describe_instances.side_effect = [instance("stopping"), instance("stopped")]
    glue.get_paginator.return_value.paginate.side_effect = [
        [{"JobRuns": []}], [{"JobRuns": [{"Id": "jr_late", "JobRunState": "STARTING"}]}], [{"JobRuns": []}],
    ]
    glue.batch_stop_job_run.return_value = {"Errors": []}
    monkeypatch.setattr(cloud.time, "sleep", lambda _: None)
    cloud.emergency_operation(s3, ec2, glue, OUT, "stop")
    glue.batch_stop_job_run.assert_called_once_with(JobName="project-etl", JobRunIds=["jr_late"])


def test_resume_clears_latch_only_after_instance_running_and_does_not_start_glue(cloud):
    events = []
    s3, ec2, glue = Mock(), Mock(), Mock()
    ec2.describe_instances.return_value = instance("stopped")
    ec2.start_instances.side_effect = lambda **_: events.append("start")
    ec2.get_waiter.return_value.wait.side_effect = lambda **_: events.append("running")
    s3.put_object.side_effect = lambda **kw: events.append(json.loads(kw["Body"])["active"])
    cloud.emergency_operation(s3, ec2, glue, OUT, "resume")
    assert events == ["start", "running", False]
    glue.start_job_run.assert_not_called()


@pytest.mark.parametrize("failure", ["start", "wait"])
def test_resume_failure_does_not_clear_latch(cloud, failure):
    s3, ec2, glue = Mock(), Mock(), Mock()
    ec2.describe_instances.return_value = instance("stopped")
    if failure == "start":
        ec2.start_instances.side_effect = RuntimeError("start failed")
    else:
        ec2.get_waiter.return_value.wait.side_effect = RuntimeError("wait failed")
    with pytest.raises(RuntimeError):
        cloud.emergency_operation(s3, ec2, glue, OUT, "resume")
    s3.put_object.assert_not_called()


@pytest.mark.parametrize("command", [
    ["glue", "--revision", "a" * 40, "--run-id", "corpus-v1"],
    ["deploy", "--release", "a" * 40, "--image", "repo@sha256:" + "a" * 64],
    ["reload-data"],
])
def test_active_latch_blocks_actual_cli_before_new_work(cloud, monkeypatch, command):
    clients = {"ssm": Mock(), "s3": control(True), "glue": Mock()}
    monkeypatch.setattr(cloud.boto3, "client", lambda name, **_: clients[name])
    monkeypatch.setattr(cloud, "outputs", lambda _: OUT)
    monkeypatch.setattr("sys.argv", ["cloud.py", "--outputs", "unused.json", *command])
    with pytest.raises(RuntimeError, match="emergência ativa"):
        cloud.main()
    clients["ssm"].send_command.assert_not_called()
    clients["glue"].start_job_run.assert_not_called()


def test_manual_emergency_does_not_wait_on_production_etl_and_uses_oidc():
    read = lambda name: yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)
    workflow = read("emergency.yml")
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["concurrency"]["group"] == "aws-emergency"
    assert "refs/heads/main" in workflow["jobs"]["emergency"]["if"]
    steps = workflow["jobs"]["emergency"]["steps"]
    auth = next(step for step in steps if "configure-aws-credentials@" in step.get("uses", ""))
    assert auth["with"]["role-to-assume"] == "${{ vars.AWS_INFRA_ROLE_ARN }}"
    for name in ("ci-cd.yml", "infrastructure.yml", "etl.yml"):
        assert "check-emergency" in (ROOT / ".github/workflows" / name).read_text()
    ci = read("ci-cd.yml")
    assert ci["on"]["push"]["branches"] == ["main"]
    assert "pull_request" not in ci["on"]
