"""Model persistence helpers with SHA256 integrity files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_hash(path: Path) -> Path:
    hash_path = path.with_suffix(path.suffix + ".sha256")
    hash_path.write_text(sha256_file(path), encoding="utf-8")
    return hash_path


def verify_hash(path: Path) -> bool:
    hash_path = path.with_suffix(path.suffix + ".sha256")
    if not hash_path.exists():
        return True
    expected = hash_path.read_text(encoding="utf-8").strip()
    return bool(expected) and sha256_file(path) == expected


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    write_hash(path)


def load_bundle(path: Path, *, verify: bool = True) -> dict[str, Any]:
    if verify and not verify_hash(path):
        raise ValueError(f"Model hash verification failed for {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Unexpected model bundle format in {path}")
    return bundle
