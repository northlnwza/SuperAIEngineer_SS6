from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


STATUS_BASE_SCORES = {
    "needs_review_total_mismatch": 120.0,
    "needs_review_missing_rows": 100.0,
    "warning_missing_page1_total": 20.0,
    "ok": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank validation_report.json documents by likely ROI for manual patching."
    )
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=ROOT / "outputs" / "debug" / "validation_report.json",
        help="Path to validation_report.json.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=30,
        help="Show up to N ranked documents.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=[],
        help="Only include these statuses. Can be passed multiple times.",
    )
    return parser.parse_args()


def doc_type_bonus(doc_id: str) -> float:
    if doc_id.startswith("party_list_"):
        return 12.0
    if doc_id.startswith("constituency_"):
        return 0.0
    return 0.0


def compute_score(item: dict) -> float:
    status = item["status"]
    expected_rows = max(int(item.get("expected_rows") or 0), 1)
    matched_rows = int(item.get("matched_rows") or 0)
    missing_rows = list(item.get("missing_row_nums") or [])

    expected_total = item.get("expected_total_votes")
    extracted_total = item.get("extracted_total_votes")
    total_gap = None
    if expected_total is not None and extracted_total is not None:
        total_gap = abs(int(expected_total) - int(extracted_total))

    score = STATUS_BASE_SCORES.get(status, 0.0)
    score += doc_type_bonus(item["doc_id"])

    if status == "needs_review_total_mismatch":
        # Favor docs with full row coverage and smaller total gaps.
        coverage_ratio = matched_rows / expected_rows
        score += coverage_ratio * 40.0
        if total_gap is not None:
            if total_gap <= 25:
                score += 60.0
            elif total_gap <= 100:
                score += 45.0
            elif total_gap <= 500:
                score += 30.0
            elif total_gap <= 2000:
                score += 15.0
            else:
                score -= 10.0
        if matched_rows == expected_rows:
            score += 25.0
    elif status == "needs_review_missing_rows":
        # Favor docs with only a few missing rows, especially if totals are close.
        missing_count = len(missing_rows)
        if missing_count <= 1:
            score += 70.0
        elif missing_count == 2:
            score += 55.0
        elif missing_count == 3:
            score += 35.0
        elif missing_count <= 5:
            score += 10.0
        else:
            score -= 30.0

        coverage_ratio = matched_rows / expected_rows
        score += coverage_ratio * 25.0

        if total_gap is not None:
            if total_gap <= 25:
                score += 35.0
            elif total_gap <= 100:
                score += 25.0
            elif total_gap <= 1000:
                score += 10.0
            else:
                score -= 10.0
    elif status == "warning_missing_page1_total":
        # These can still be worth reviewing when all rows are present.
        if matched_rows == expected_rows:
            score += 15.0

    return round(score, 2)


def priority_label(score: float) -> str:
    if score >= 200:
        return "high"
    if score >= 140:
        return "medium"
    return "low"


def reason_text(item: dict) -> str:
    status = item["status"]
    missing_rows = list(item.get("missing_row_nums") or [])
    expected_total = item.get("expected_total_votes")
    extracted_total = item.get("extracted_total_votes")
    total_gap = None
    if expected_total is not None and extracted_total is not None:
        total_gap = abs(int(expected_total) - int(extracted_total))

    if status == "needs_review_total_mismatch":
        return (
            f"full_rows={item['matched_rows'] == item['expected_rows']}, "
            f"total_gap={total_gap}"
        )
    if status == "needs_review_missing_rows":
        return (
            f"missing_count={len(missing_rows)}, "
            f"missing_rows={missing_rows}, "
            f"total_gap={total_gap}"
        )
    return f"matched={item['matched_rows']}/{item['expected_rows']}"


def main() -> None:
    args = parse_args()
    data = json.loads(args.validation_json.read_text(encoding="utf-8"))

    items = list(data.values())
    if args.status:
        allowed = set(args.status)
        items = [item for item in items if item["status"] in allowed]
    else:
        items = [item for item in items if item["status"] != "ok"]

    ranked = []
    for item in items:
        score = compute_score(item)
        ranked.append(
            {
                "doc_id": item["doc_id"],
                "status": item["status"],
                "score": score,
                "priority": priority_label(score),
                "matched_rows": item["matched_rows"],
                "expected_rows": item["expected_rows"],
                "missing_row_nums": item["missing_row_nums"],
                "expected_total_votes": item["expected_total_votes"],
                "extracted_total_votes": item["extracted_total_votes"],
                "reason": reason_text(item),
            }
        )

    ranked.sort(
        key=lambda item: (
            -item["score"],
            item["status"],
            item["doc_id"],
        )
    )

    print("patch_queue_count", len(ranked))
    for item in ranked[: args.show]:
        print(
            item["priority"],
            f"score={item['score']}",
            item["doc_id"],
            item["status"],
            f"matched={item['matched_rows']}/{item['expected_rows']}",
            f"total={item['extracted_total_votes']}/{item['expected_total_votes']}",
            item["reason"],
        )


if __name__ == "__main__":
    main()
