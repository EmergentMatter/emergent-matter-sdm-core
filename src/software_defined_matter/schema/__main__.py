"""CLI: print an ``.sdm`` JSON Schema, or the version index, by version.

    python -m software_defined_matter.schema --version 0.3
    python -m software_defined_matter.schema --index

This is the subprocess boundary a host without this package's full
dependency set (no JAX, no jsonschema) uses to fetch the wire contract.
Importing ``software_defined_matter.schema`` alone never pulls in either.
``--index`` reads the committed ``index.json`` directly rather than
regenerating it, so it stays that same no-import-of-core boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from software_defined_matter.io import load_schema
from software_defined_matter.model import KNOWN_SCHEMA_VERSIONS

_INDEX_PATH = Path(__file__).parent / "index.json"


def _load_index() -> dict[str, Any]:
    with _INDEX_PATH.open() as fh:
        result: dict[str, Any] = json.load(fh)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m software_defined_matter.schema",
        description="Print the .sdm JSON Schema for a given schema_version, or the version index.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--version",
        help=f"schema_version to print. Known versions: {KNOWN_SCHEMA_VERSIONS}.",
    )
    group.add_argument(
        "--index",
        action="store_true",
        help="Print the machine-readable version index (which versions exist, "
        "which is newest, and what each added).",
    )
    args = parser.parse_args(argv)

    if args.index:
        try:
            index = _load_index()
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(index, indent=2))
        return 0

    try:
        schema = load_schema(args.version)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(schema, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
