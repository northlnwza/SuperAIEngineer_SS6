from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect suspicious rows inside parsed_rows.json documents."
    )
    parser.add_argument(
        "--parsed-json",
        type=Path,
        default=ROOT / "outputs" / "debug" / "parsed_rows.json",
        help="Path to parsed_rows.json.",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        default=[],
        help="Document id to inspect. Can be passed multiple times.",
    )
    parser.add_argument(
        "--top-from-rank",
        type=int,
        default=0,
        help="Automatically inspect top N docs from the validation-based ranking.",
    )
    parser.add_argument(
        "--show-rows",
        type=int,
        default=12,
        help="Max suspicious rows to print per document.",
    )
    return parser.parse_args()


def load_ranked_doc_ids(debug_data: dict[str, dict], top_n: int) -> list[str]:
    def score_doc(doc: dict) -> float:
        validation = doc.get("validation", {})
        status = validation.get("status", "ok")
        if status == "ok":
            return -1.0
        expected_rows = max(int(validation.get("expected_rows") or 0), 1)
        matched_rows = int(validation.get("matched_rows") or 0)
        expected_total = validation.get("expected_total_votes")
        extracted_total = validation.get("extracted_total_votes")
        total_gap = (
            abs(int(expected_total) - int(extracted_total))
            if expected_total is not None and extracted_total is not None
            else None
        )
        missing_count = len(validation.get("missing_row_nums") or [])

        base = {
            "needs_review_total_mismatch": 100.0,
            "needs_review_missing_rows": 80.0,
            "warning_missing_page1_total": 10.0,
        }.get(status, 0.0)
        if matched_rows == expected_rows:
            base += 25.0
        if total_gap is not None:
            if total_gap <= 25:
                base += 50.0
            elif total_gap <= 100:
                base += 35.0
            elif total_gap <= 500:
                base += 20.0
        if missing_count <= 2:
            base += 30.0
        elif missing_count <= 4:
            base += 10.0
        return base

    ranked = sorted(
        debug_data.items(),
        key=lambda item: (-score_doc(item[1]), item[0]),
    )
    return [
        doc_id for doc_id, doc in ranked
        if doc.get("validation", {}).get("status") != "ok"
    ][:top_n]


def suspicious_rows_for_doc(doc: dict) -> list[tuple[float, dict, list[str]]]:
    assigned_rows = doc.get("assigned_rows", [])
    page_total_flags = doc.get("page_total_flags", {})
    expected_total = doc.get("expected_total_votes")

    results: list[tuple[float, dict, list[str]]] = []
    for row in assigned_rows:
        reasons: list[str] = []
        score = 0.0
        votes_text = str(row.get("votes") or "")
        vote_words_value = row.get("vote_words_value")
        source = str(row.get("matched_source") or "")
        matched = bool(row.get("matched"))
        rescued = row.get("rescued")
        row_num = int(row.get("row_num") or 0)

        if not matched:
            reasons.append("unmatched")
            score += 100.0
        if rescued:
            reasons.append(f"rescued={rescued}")
            score += 35.0
        if page_total_flags.get(source, False):
            reasons.append("on_total_row_page")
            score += 10.0
        if votes_text.isdigit() and vote_words_value is not None and int(votes_text) != int(vote_words_value):
            diff = abs(int(votes_text) - int(vote_words_value))
            reasons.append(f"digit_vs_words={votes_text}/{vote_words_value}")
            score += 30.0
            if diff <= 25:
                score += 20.0
            elif diff <= 100:
                score += 10.0
        if votes_text.isdigit() and expected_total is not None and int(votes_text) == int(expected_total):
            reasons.append("equals_doc_total")
            score += 50.0
        if row_num >= 55:
            reasons.append("tail_row")
            score += 5.0

        if reasons:
            results.append((score, row, reasons))

    results.sort(key=lambda item: (-item[0], int(item[1].get("row_num") or 0)))
    return results


def print_doc(doc_id: str, doc: dict, show_rows: int) -> None:
    validation = doc.get("validation", {})
    print(
        f"\nDOC {doc_id} status={validation.get('status')} "
        f"matched={validation.get('matched_rows')}/{validation.get('expected_rows')} "
        f"total={validation.get('extracted_total_votes')}/{validation.get('expected_total_votes')}"
    )
    if validation.get("missing_row_nums"):
        print(
            "missing_rows",
            validation.get("missing_row_nums"),
            "missing_parties",
            validation.get("missing_parties"),
        )

    suspicious = suspicious_rows_for_doc(doc)
    if not suspicious:
        print("no_suspicious_rows_found")
        return

    print("suspicious_rows")
    for score, row, reasons in suspicious[:show_rows]:
        print(
            f"  row={row.get('row_num')} party={row.get('party_name')} "
            f"votes={row.get('votes')} words={row.get('vote_words_value')} "
            f"source={row.get('matched_source')} score={round(score, 2)} "
            f"reasons={'; '.join(reasons)}"
        )


def main() -> None:
    args = parse_args()
    debug_data = json.loads(args.parsed_json.read_text(encoding="utf-8"))

    doc_ids = list(args.doc_id)
    if args.top_from_rank:
        ranked_ids = load_ranked_doc_ids(debug_data, args.top_from_rank)
        for doc_id in ranked_ids:
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)

    if not doc_ids:
        raise SystemExit("Provide at least one --doc-id or use --top-from-rank.")

    for doc_id in doc_ids:
        doc = debug_data.get(doc_id)
        if doc is None:
            print(f"\nDOC {doc_id} not_found")
            continue
        print_doc(doc_id, doc, args.show_rows)


if __name__ == "__main__":
    main()
