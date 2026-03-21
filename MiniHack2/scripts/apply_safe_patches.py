from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply high-confidence Thai-word patch suggestions to a submission CSV."
    )
    parser.add_argument(
        "--parsed-json",
        type=Path,
        default=ROOT / "outputs" / "debug" / "parsed_rows.json",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "outputs" / "submissions" / "submission_kaggle.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "outputs" / "submissions" / "submission_kaggle_safe_patched.csv",
    )
    parser.add_argument(
        "--max-changes",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-delta",
        type=int,
        default=20,
        help="Largest absolute row-value change allowed in a safe patch.",
    )
    parser.add_argument(
        "--min-improvement",
        type=int,
        default=5,
        help="Minimum total-gap reduction required before applying a patch set.",
    )
    return parser.parse_args()


def load_submission_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_submission_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "votes"])
        writer.writeheader()
        writer.writerows(rows)


def best_patch_set(doc: dict, max_changes: int) -> tuple[int, int, list[dict]] | None:
    validation = doc.get("validation", {})
    expected = validation.get("expected_total_votes")
    extracted = validation.get("extracted_total_votes")
    if expected is None or extracted is None:
        return None
    expected = int(expected)
    extracted = int(extracted)
    gap_before = abs(expected - extracted)

    candidates = []
    for row in doc.get("assigned_rows", []):
        votes = row.get("votes")
        word_value = row.get("vote_words_value")
        if not (isinstance(votes, str) and votes.isdigit()):
            continue
        if word_value is None:
            continue
        current = int(votes)
        suggested = int(word_value)
        if current == suggested:
            continue
        if suggested == expected:
            continue
        candidates.append(
            {
                "row_num": int(row["row_num"]),
                "party_name": row["party_name"],
                "current": current,
                "suggested": suggested,
                "delta": suggested - current,
            }
        )

    if not candidates:
        return None

    best_gap = gap_before
    best_combo: list[dict] = []
    for count in range(1, min(max_changes, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, count):
            trial = extracted + sum(item["delta"] for item in combo)
            gap = abs(expected - trial)
            if gap < best_gap:
                best_gap = gap
                best_combo = list(combo)
                if gap == 0:
                    return gap_before, best_gap, best_combo
    if not best_combo:
        return None
    return gap_before, best_gap, best_combo


def is_safe_patch(result: tuple[int, int, list[dict]], max_delta: int, min_improvement: int) -> bool:
    gap_before, gap_after, combo = result
    if gap_before - gap_after < min_improvement:
        return False
    if gap_after > gap_before * 0.5 and gap_after > 10:
        return False
    if any(abs(item["delta"]) > max_delta for item in combo):
        return False
    return True


def main() -> None:
    args = parse_args()
    data = json.loads(args.parsed_json.read_text(encoding="utf-8"))
    rows = load_submission_rows(args.input_csv)
    rows_by_doc_row = {}
    for row in rows:
        doc_id, row_num = row["id"].rsplit("_", 1)
        rows_by_doc_row[(doc_id, int(row_num))] = row

    applied = []
    for doc_id, doc in data.items():
        validation = doc.get("validation", {})
        if validation.get("status") != "needs_review_total_mismatch":
            continue
        result = best_patch_set(doc, args.max_changes)
        if result is None or not is_safe_patch(result, args.max_delta, args.min_improvement):
            continue
        gap_before, gap_after, combo = result
        for item in combo:
            row = rows_by_doc_row.get((doc_id, item["row_num"]))
            if row is not None:
                row["votes"] = str(item["suggested"])
        applied.append(
            {
                "doc_id": doc_id,
                "gap_before": gap_before,
                "gap_after": gap_after,
                "patches": combo,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_submission_rows(args.output_csv, rows)

    print(f"wrote={args.output_csv}")
    print(f"applied_docs={len(applied)}")
    for item in applied[:20]:
        print(
            item["doc_id"],
            f"gap={item['gap_before']}->{item['gap_after']}",
            "patches=",
            [
                f"row {patch['row_num']} {patch['current']}->{patch['suggested']}"
                for patch in item["patches"]
            ],
        )


if __name__ == "__main__":
    main()
