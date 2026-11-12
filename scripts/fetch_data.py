"""Download FreshRetailNet-50K at a pinned revision and record checksums.

Usage: python scripts/fetch_data.py [--config configs/small.yaml] [--verify]
"""

from __future__ import annotations

import argparse
import json

from sctwin.data import fetch, verify_checksums
from sctwin.repro import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--verify", action="store_true", help="only re-check local checksums")
    a = ap.parse_args()
    cfg = load_config(a.config)["dataset"]
    if a.verify:
        res = verify_checksums(cfg["data_dir"])
        print(json.dumps(res, indent=2))
        raise SystemExit(0 if all(res.values()) else 1)
    prov = fetch(cfg["data_dir"], revision=cfg.get("revision"), repo_id=cfg["repo_id"])
    print(json.dumps(prov, indent=2))
    if not cfg.get("revision"):
        print(f"\nPin this in the config:  dataset.revision: {prov['revision']}")


if __name__ == "__main__":
    main()
