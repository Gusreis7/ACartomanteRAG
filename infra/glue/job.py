"""Entrypoint Glue 5.1: ambiente isolado e módulos iguais aos do projeto."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import zipfile

import boto3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets-bucket", required=True)
    p.add_argument("--bundle-prefix", required=True)
    p.add_argument("--settings-parameter", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--steps", default="crawl,bronze,silver,text-index")
    p.add_argument("--reuse-index-id", default="")
    p.add_argument("--promote", default="false", choices=["true", "false"])
    p.add_argument("--smoke", default="false", choices=["true", "false"])
    args, _ = p.parse_known_args()
    region = os.getenv("AWS_REGION", "us-east-1")
    settings = json.loads(boto3.client("ssm", region_name=region).get_parameter(Name=args.settings_parameter, WithDecryption=True)["Parameter"]["Value"])
    # ETL usa uma chave Qdrant de escrita; a aplicação usa outra, somente leitura.
    os.environ.update(QDRANT_URL=settings["QDRANT_URL"], QDRANT_API_KEY=settings["QDRANT_API_KEY"])
    with tempfile.TemporaryDirectory(prefix="acartomante-") as tmp:
        root = Path(tmp)
        archive = root / "source.zip"
        s3 = boto3.client("s3", region_name=region)
        s3.download_file(args.assets_bucket, f"{args.bundle_prefix}/source.zip", str(archive))
        with zipfile.ZipFile(archive) as z:
            for item in z.namelist():
                path = Path(item)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Caminho inválido no bundle de fontes")
            z.extractall(root)
        bucket_map = json.loads(s3.get_object(Bucket=args.assets_bucket, Key="config/buckets.json")["Body"].read())
        os.environ.update(bucket_map)
        os.environ.update(
            STORAGE_PROVIDER="s3", CONFIG_FILE=str(root / "config/config.aws.yaml"),
            ENV_FILE=str(root / "config/absent.env"), HF_HOME=str(root / "hf-cache"),
            SOURCE_REVISION=(root / "config/source_revision.txt").read_text().strip(),
        )
        # A lib do driver empacotada no job é a mesma revisão baixada acima.
        sys.path.insert(0, str(root))
        if args.smoke == "true":
            from src.config.config_manager import ConfigManager
            from src.embeddings.registry import EmbedderRegistry
            from src.etl.production import validate_production_config
            from src.generation.provider import create_generation_provider
            from src.storage.rustfs_manager import RustFSManager
            from src.storage.qdrant_manager import QdrantManager
            import pymupdf
            config = ConfigManager.load()
            validate_production_config(config)
            storage = RustFSManager.from_config(config.environment.rustfs)
            for field, bucket in config.storage.model_dump().items():
                if not field.endswith("_bucket"):
                    continue
                if not storage.bucket_exists(bucket):
                    raise ValueError("Bucket necessário ausente")
            qdrant = QdrantManager.from_config(config.environment.qdrant)
            try:
                qdrant.info()
            finally:
                qdrant.close()
            create_generation_provider(config.generation).healthcheck()
            embedder = EmbedderRegistry().create(config.embedding.models["colibri"], normalize_embeddings=config.embedding.index.normalize_embeddings, device="cpu", batch_size=1)
            vectors = embedder.embed_documents(["Machado de Assis escreveu A Cartomante."])
            if len(vectors) != 1:
                raise ValueError("Smoke de embeddings falhou")
            print("[ok] Glue, PyMuPDF, S3, Qdrant, catálogo Bedrock e Colibri; nenhuma geração LLM executada.")
            return
        sys.argv = ["src.etl.production", "--run-id", args.run_id, "--steps", *args.steps.split(",")]
        if args.reuse_index_id:
            sys.argv += ["--reuse-index-id", args.reuse_index_id]
        if args.promote == "true":
            sys.argv += ["--promote"]
        runpy.run_module("src.etl.production", run_name="__main__")


if __name__ == "__main__":
    main()
