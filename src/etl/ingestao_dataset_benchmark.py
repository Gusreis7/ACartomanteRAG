from pathlib import Path

from src.config.config_manager import ConfigManager
from src.storage.rustfs_manager import RustFSManager

OBJECT_KEY = "retrieval/v1/benchmark_questions.csv"
FILE_PATH = Path("AED/benchmark_questions.csv")


def main() -> None:
    config = ConfigManager.load()

    rustfs = RustFSManager.from_config(config.environment.rustfs)

    bucket = config.storage.benchmarks_bucket

    if not rustfs.bucket_exists(bucket):
        rustfs.create_bucket(bucket)

    rustfs.upload_file(
        bucket=bucket,
        key=OBJECT_KEY,
        local_path=FILE_PATH,
    )

    print(f"[ok] s3://{bucket}/{OBJECT_KEY}")


if __name__ == "__main__":
    main()
