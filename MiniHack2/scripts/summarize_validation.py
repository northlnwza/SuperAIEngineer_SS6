from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize validation_report.json.")
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=ROOT / "outputs" / "debug" / "validation_report.json",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="Show up to N non-ok documents.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.validation_json.read_text(encoding="utf-8"))
    counts = Counter(item["status"] for item in data.values())
    print("status_counts", dict(counts))

    flagged = [
        item for item in data.values()
        if item["status"] != "ok"
    ]
    flagged.sort(key=lambda item: (item["status"], item["doc_id"]))
    for item in flagged[: args.show]:
        print(
            item["doc_id"],
            item["status"],
            f"matched={item['matched_rows']}/{item['expected_rows']}",
            f"total={item['extracted_total_votes']}/{item['expected_total_votes']}",
            f"missing_rows={item['missing_row_nums']}",
        )


if __name__ == "__main__":
    main()
