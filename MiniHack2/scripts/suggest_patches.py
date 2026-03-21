from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest row-value substitutions from Thai number words to reduce total gaps."
    )
    parser.add_argument(
        "--parsed-json",
        type=Path,
        default=ROOT / "outputs" / "debug" / "parsed_rows.json",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        default=[],
        help="Specific doc_id to inspect. Can be passed multiple times.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="If no doc_id is given, inspect the top N mismatch docs by smallest total gap.",
    )
    parser.add_argument(
        "--max-changes",
        type=int,
        default=3,
        help="Maximum number of row substitutions to consider per document.",
    )
    return parser.parse_args()


def load_docs(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_docs(data: dict[str, dict], doc_ids: list[str], top: int) -> list[str]:
    if doc_ids:
        return doc_ids

    mismatch_docs: list[tuple[int, str]] = []
    for doc_id, doc in data.items():
        validation = doc.get("validation", {})
        if validation.get("status") != "needs_review_total_mismatch":
            continue
        expected = validation.get("expected_total_votes")
        extracted = validation.get("extracted_total_votes")
        if expected is None or extracted is None:
            continue
        gap = abs(int(expected) - int(extracted))
        mismatch_docs.append((gap, doc_id))
    mismatch_docs.sort()
    return [doc_id for _, doc_id in mismatch_docs[:top]]


def candidate_rows(doc: dict) -> list[dict]:
    rows = []
    expected_total = doc.get("expected_total_votes")
    for row in doc.get("assigned_rows", []):
        votes = row.get("votes")
        word_value = row.get("vote_words_value")
        if not (isinstance(votes, str) and votes.isdigit()):
            continue
        if word_value is None:
            continue
        word_value = int(word_value)
        vote_value = int(votes)
        if word_value == vote_value:
            continue
        # Avoid obvious document-total leakage from the Thai number words.
        if expected_total is not None and word_value == int(expected_total):
            continue
        rows.append(
            {
                "row_num": int(row["row_num"]),
                "party_name": row["party_name"],
                "current": vote_value,
                "suggested": word_value,
                "delta": word_value - vote_value,
                "source": row.get("matched_source"),
            }
        )
    return rows


def current_total(doc: dict) -> int | None:
    validation = doc.get("validation", {})
    extracted = validation.get("extracted_total_votes")
    return None if extracted is None else int(extracted)


def expected_total(doc: dict) -> int | None:
    validation = doc.get("validation", {})
    expected = validation.get("expected_total_votes")
    return None if expected is None else int(expected)


def best_patch_set(doc: dict, max_changes: int) -> tuple[int, list[dict], int] | None:
    current = current_total(doc)
    expected = expected_total(doc)
    if current is None or expected is None:
        return None

    gap_before = abs(expected - current)
    candidates = candidate_rows(doc)
    if not candidates:
        return None

    best_gap = gap_before
    best_combo: list[dict] = []
    best_total = current

    limit = min(max_changes, len(candidates))
    for change_count in range(1, limit + 1):
        for combo in itertools.combinations(candidates, change_count):
            trial_total = current + sum(item["delta"] for item in combo)
            gap = abs(expected - trial_total)
            if gap < best_gap:
                best_gap = gap
                best_combo = list(combo)
                best_total = trial_total
                if gap == 0:
                    return best_gap, best_combo, best_total

    if not best_combo:
        return None
    return best_gap, best_combo, best_total


def main() -> None:
    args = parse_args()
    data = load_docs(args.parsed_json)
    doc_ids = choose_docs(data, args.doc_id, args.top)

    for doc_id in doc_ids:
        doc = data.get(doc_id)
        if doc is None:
            print(f"\nDOC {doc_id} not_found")
            continue
        validation = doc.get("validation", {})
        before = abs(
            int(validation["expected_total_votes"]) - int(validation["extracted_total_votes"])
        )
        result = best_patch_set(doc, args.max_changes)
        print(
            f"\nDOC {doc_id} status={validation.get('status')} "
            f"matched={validation.get('matched_rows')}/{validation.get('expected_rows')} "
            f"total={validation.get('extracted_total_votes')}/{validation.get('expected_total_votes')} "
            f"gap_before={before}"
        )
        if result is None:
            print("  no_safe_word_value_patch_found")
            continue
        gap_after, combo, trial_total = result
        print(f"  suggested_total={trial_total} gap_after={gap_after}")
        for item in combo:
            print(
                f"  patch row={item['row_num']} party={item['party_name']} "
                f"{item['current']} -> {item['suggested']} delta={item['delta']} "
                f"source={item['source']}"
            )


if __name__ == "__main__":
    main()
