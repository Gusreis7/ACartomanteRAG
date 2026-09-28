import argparse
from pathlib import Path
from src.config.config_manager import ConfigManager
from src.indexing.release_manager import ReleaseManager
from src.storage.rustfs_manager import RustFSManager

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-id", required=True)
    p.add_argument("--release", required=True)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--config-file", type=Path, default=None)
    a = p.parse_args()
    kw = {}
    if a.env_file is not None: kw["env_file"] = a.env_file
    if a.config_file is not None: kw["yaml_file"] = a.config_file
    config = ConfigManager.load(**kw)
    rustfs = RustFSManager.from_config(config.environment.rustfs)
    pointer = ReleaseManager(config, rustfs).publish(a.index_id, a.release)
    print(f"release: {pointer.release}")
    print(f"index_id: {pointer.index_id}")
    print(f"published_at_utc: {pointer.published_at_utc}")

if __name__ == "__main__":
    main()
