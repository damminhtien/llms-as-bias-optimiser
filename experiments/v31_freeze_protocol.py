"""Verify the V3.1 protocol and its frozen V2/V3 provenance before a run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bias_optimizer.evaluation.protocol import load_frozen_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    protocol, manifest = load_frozen_protocol(args.root)
    print(
        json.dumps(
            {
                "study_id": protocol["study_id"],
                "frozen_on": protocol["frozen_on"],
                "protocol_sha256": manifest["protocol"]["sha256"],
                "v3_finalists": len(manifest["v3_finalist_ast_sha256"]),
                "protected_sources_verified": len(manifest["protected_source_files"]),
                "v3_search_enabled": protocol["v3_search_enabled"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
