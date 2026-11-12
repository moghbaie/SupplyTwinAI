"""Reproducibility helpers: seeding, config loading/hashing, run manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TRACKED_PACKAGES = (
    "numpy",
    "polars",
    "pyarrow",
    "pydantic",
    "huggingface_hub",
    "lightgbm",
    "scipy",
    "jax",
    "numpyro",
    "arviz",
)


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python and NumPy global state; return a fresh Generator for explicit use."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global state, for third-party code only
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} is not a mapping")
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable short hash of a config (key order independent)."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _pkg_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in TRACKED_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": _pkg_versions(),
        "colab_runtime": os.environ.get("COLAB_RELEASE_TAG"),
    }
    try:
        import jax  # type: ignore

        info["accelerators"] = [str(d) for d in jax.devices()]
    except Exception:  # noqa: BLE001 - jax optional
        info["accelerators"] = None
    return info


def write_run_manifest(out_dir: str | Path, cfg: dict[str, Any], **extra: Any) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "environment": environment_info(),
        **extra,
    }
    path = out / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
