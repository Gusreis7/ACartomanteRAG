"""Fixa revisões de tokenizer, Colibri e reranker antes de empacotar o release."""
import json
from pathlib import Path
import re
from urllib.request import urlopen
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    path = ROOT / "config/config.aws.yaml"
    overlay = yaml.safe_load(path.read_text())
    selected = [
        ("silver", "tokenizers", base["silver"]["chunking"]["tokenizer"]),
        ("embedding", "models", "colibri"),
        ("retrievals", "dense_cross_encoder"),
    ]
    for route in selected:
        source = base
        target = overlay
        for key in route:
            source = source[key]
            target = target.setdefault(key, {})
        revision = target.get("revision", source.get("revision", "main"))
        if re.fullmatch(r"[0-9a-f]{40}", revision):
            continue
        model = source["model"]
        with urlopen(f"https://huggingface.co/api/models/{model}/revision/{revision}", timeout=30) as response:
            sha = json.load(response)["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"Revisão não imutável para {model}")
        target["revision"] = sha
        print(f"{model}: {sha}")
    path.write_text(yaml.safe_dump(overlay, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
