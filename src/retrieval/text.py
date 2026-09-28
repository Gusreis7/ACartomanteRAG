from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(
        char for char in value if not unicodedata.combining(char)
    )
    value = value.casefold()
    return " ".join(TOKEN_RE.findall(value))


def tokenize(value: str) -> list[str]:
    normalized = normalize_text(value)
    return normalized.split() if normalized else []
