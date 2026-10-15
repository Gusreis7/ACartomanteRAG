"""Importa o cluster real, gerando configuração fiel e recusando alterações remotas."""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--account-id", required=True)
    p.add_argument("--cluster-id", required=True)
    p.add_argument("--apply-import", action="store_true")
    args=p.parse_args()
    uuid_pattern=r"[0-9a-fA-F-]{36}"
    if not re.fullmatch(uuid_pattern,args.account_id) or not re.fullmatch(uuid_pattern,args.cluster_id):
        p.error("Informe IDs UUID reais da conta e do cluster")
    folder=ROOT/"infra/qdrant"
    backend=ROOT/"infra/aws/backend.hcl"
    if not backend.is_file():raise FileNotFoundError("Prepare o backend AWS primeiro")
    (folder/"account.auto.tfvars.json").write_text(json.dumps({"account_id":args.account_id}))
    env={**os.environ, "TF_VAR_account_id":args.account_id,
         "TF_VAR_cloud_api_key":getpass.getpass("Qdrant CLOUD API key (administração, não a chave de banco): ")}
    def tf(*argv,capture=False):
        return subprocess.run(["terraform",f"-chdir={folder}",*argv],env=env,text=True,capture_output=capture,check=True)
    tf("init",f"-backend-config={backend}","-backend-config=key=acartomanterag/qdrant.tfstate","-input=false")
    imports=folder/"import.tf"
    imports.write_text(f'import {{\n  to = qdrant-cloud_accounts_cluster.existing\n  id = "{args.cluster_id}"\n}}\n')
    generated=folder/"cluster.generated.tf"
    if not generated.exists():
        tf("plan","-generate-config-out=cluster.generated.tf","-input=false","-out=import.tfplan")
        text=generated.read_text()
        start='resource "qdrant-cloud_accounts_cluster" "existing" {'
        if start not in text:raise ValueError("Formato da configuração gerada inesperado")
        generated.write_text(text.replace(start,start+'\n  lifecycle { prevent_destroy = true }',1))
        tf("fmt")
    tf("plan","-input=false","-out=import.tfplan")
    plan=json.loads(tf("show","-json","import.tfplan",capture=True).stdout)
    for change in plan.get("resource_changes",[]):
        if change.get("change",{}).get("actions") != ["no-op"]:
            raise ValueError("O plano propõe alteração no cluster; importação interrompida. Revise cluster.generated.tf.")
    if args.apply_import:
        tf("apply","-input=false","import.tfplan")
        imports.unlink()
        print("Cluster importado sem alteração remota. Commitar cluster.generated.tf e o lock do provider; manter o estado privado.")
    else:
        print("Plano sem alterações remotas. Confira import.tfplan e repita com --apply-import para persistir o import.")


if __name__ == "__main__":main()
