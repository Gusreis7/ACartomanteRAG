"""Exercita os scripts locais sem Docker Engine, downloads ou inferência."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RaspberryScriptsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="raspberry-test-")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.bin = self.work / "bin"
        self.bin.mkdir()
        self.env = {"PATH": str(self.bin) + os.pathsep + os.defpath, "HOME": str(self.work)}

    def executable(self, name, text):
        target = self.bin / name
        target.write_text(text)
        target.chmod(0o755)
        return target

    def fake_ollama(self):
        self.env.update(OLLAMA_MODEL="qwen3.5:4b", OLLAMA_HOST="http://ollama:11434", STATE=str(self.work))
        self.executable("ollama", """#!/bin/sh
set -eu
printf '%s %s\n' "$1" "${2:-}" >> "$STATE/calls"
case "$1" in
  show) test -f "$STATE/model" ;;
  pull)
    if [ -f "$STATE/fail" ]; then echo 'download interrupted' >&2; exit 7; fi
    if [ ! -f "$STATE/no-model" ]; then touch "$STATE/model"; fi
    ;;
  *) exit 9 ;;
esac
""")

    def prepare(self, **extra):
        return subprocess.run(["sh", str(ROOT / "scripts/prepare_ollama.sh")], env={**self.env, **extra}, text=True, capture_output=True)

    def calls(self):
        return (self.work / "calls").read_text().splitlines()

    def test_existing_model_does_not_download(self):
        self.fake_ollama()
        (self.work / "model").touch()
        result = self.prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), ["show qwen3.5:4b"])

    def test_download_is_verified_and_reused(self):
        self.fake_ollama()
        result = self.prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), ["show qwen3.5:4b", "pull qwen3.5:4b", "show qwen3.5:4b"])
        self.assertEqual(self.prepare().returncode, 0)
        self.assertEqual(self.calls().count("pull qwen3.5:4b"), 1)

    def test_failed_download_returns_failure_and_detail(self):
        self.fake_ollama()
        (self.work / "fail").touch()
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("download interrupted", result.stderr)
        self.assertIn("API não será iniciada", result.stderr)
        self.assertFalse((self.work / "model").exists())

    def test_successful_pull_without_model_is_rejected(self):
        self.fake_ollama()
        (self.work / "no-model").touch()
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("não ficou disponível", result.stderr)

    def test_missing_model_is_rejected_before_cli(self):
        self.fake_ollama()
        self.env.pop("OLLAMA_MODEL")
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.work / "calls").exists())

    def fake_docker(self):
        self.env["TRACE"] = str(self.work / "docker.jsonl")
        python = shutil.which("python3")
        self.executable("docker", f"""#!{python}
import json, os, sys
with open(os.environ['TRACE'], 'a') as f:
    f.write(json.dumps({{'args': sys.argv[1:], 'bind': os.getenv('API_BIND_IP')}})+'\\n')
""")

    def trace(self):
        return [json.loads(line) for line in Path(self.env["TRACE"]).read_text().splitlines()]

    def test_wrapper_quotes_paths_and_uses_existing_private_env(self):
        self.fake_docker()
        project = self.work / "projeto com espaços"
        (project / "scripts").mkdir(parents=True)
        (project / "config").mkdir()
        (project / "config/.env").write_text("API_BIND_IP=192.0.2.15\n")
        shutil.copyfile(ROOT / "scripts/compose_raspberry.sh", project / "scripts/compose_raspberry.sh")
        result = subprocess.run(["bash", str(project / "scripts/compose_raspberry.sh"), "config", "--quiet"], env=self.env, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.trace()[0]["args"]
        self.assertEqual(args[args.index("--env-file") + 1], str(project / "config/.env"))
        self.assertIn(str(project / "docker-compose.raspberry.yml"), args)

    def test_wrapper_only_defaults_bind_in_temporary_ci(self):
        self.fake_docker()
        script = str(ROOT / "scripts/compose_raspberry.sh")
        for extra, expected in [({}, None), ({"COMPOSE_PROJECT_NAME": "acartomante-ci-42"}, "127.0.0.1"), ({"COMPOSE_PROJECT_NAME": "acartomante-ci-42", "API_BIND_IP": "192.0.2.12"}, "192.0.2.12")]:
            result = subprocess.run(["bash", script, "config", "--quiet"], env={**self.env, **extra}, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.trace()[-1]["bind"], expected)

    def test_explicit_missing_env_is_an_error(self):
        self.fake_docker()
        result = subprocess.run(["bash", str(ROOT / "scripts/compose_raspberry.sh"), "config"], env={**self.env, "LOCAL_ENV_FILE": str(self.work / "missing.env")}, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(Path(self.env["TRACE"]).exists())

    def test_maintenance_check_does_not_touch_production_services(self):
        self.fake_docker()
        result = subprocess.run(["bash", str(ROOT / "scripts/check_raspberry_ci.sh")], env={**self.env, "COMPOSE_PROJECT_NAME": "acartomante-prod"}, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(args["args"][-2:] == ["config", "--quiet"] for args in self.trace()))

    def test_maintenance_check_only_parses_compose_without_secrets(self):
        self.fake_docker()
        result = subprocess.run(["bash", str(ROOT / "scripts/check_raspberry_ci.sh")], env={**self.env, "COMPOSE_PROJECT_NAME": "acartomante-ci-42"}, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [line["args"] for line in self.trace()]
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(args[:1] == ["compose"] and args[-2:] == ["config", "--quiet"] for args in calls))
        self.assertIn(b"deploy do Raspberry suspensos", result.stdout)


if __name__ == "__main__":
    unittest.main()
