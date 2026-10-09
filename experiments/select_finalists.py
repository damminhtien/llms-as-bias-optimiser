"""Select and freeze low-data finalists before final test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bias_optimizer.search.selection import select_finalists


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("results/search.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/finalists.json"),
    )
    args = parser.parse_args()
    metadata = select_finalists(archive_path=args.archive, output_path=args.output)
    print(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
