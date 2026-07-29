"""Strict, deterministic artifact helpers shared by P4 and later phases."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch


def validate_sha256(value: Any, *, name: str) -> str:
    """Return a normalized SHA-256 digest or fail closed."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal characters")
    return normalized


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in a stable order."""

    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict[{name!r}] must be a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def strict_json_load(path: str | Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


def publish_staged_files(
    staged_by_destination: Mapping[str | Path, str | Path],
    *,
    overwrite: bool = False,
) -> None:
    """Publish a small staged bundle with rollback on ordinary I/O failures."""

    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    normalized = {
        Path(destination): Path(staged)
        for destination, staged in staged_by_destination.items()
    }
    if not normalized:
        raise ValueError("at least one staged file is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("bundle destinations must be unique")
    for destination, staged in normalized.items():
        if destination == staged:
            raise ValueError("staged and destination paths must differ")
        if not staged.is_file():
            raise FileNotFoundError(staged)
        destination.parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in normalized if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "artifact bundle already exists: "
            + ", ".join(str(path) for path in existing)
        )

    token = uuid4().hex
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for destination in existing:
            backup = destination.with_name(f".{destination.name}.{token}.backup")
            os.replace(destination, backup)
            backups[destination] = backup
        for destination, staged in normalized.items():
            os.replace(staged, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            if destination.is_file():
                destination.unlink()
        for destination, backup in backups.items():
            if backup.is_file():
                os.replace(backup, destination)
        raise
    else:
        for backup in backups.values():
            if backup.is_file():
                backup.unlink()
    finally:
        for staged in normalized.values():
            if staged.is_file():
                staged.unlink()


def strict_json_write(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    staged.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    publish_staged_files({output: staged}, overwrite=True)


__all__ = [
    "canonical_json_bytes",
    "canonical_json_sha256",
    "publish_staged_files",
    "sha256_file",
    "state_dict_sha256",
    "strict_json_load",
    "strict_json_write",
    "validate_sha256",
]
