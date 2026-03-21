from __future__ import annotations

from dataclasses import dataclass
import re

from .normalize import extract_vote_digits


SECTION_41_RE = re.compile(
    r"(?:4|๔)\\?\.(?:1|๑)[^\n]*?([0-9๐-๙][0-9๐-๙,]*)",
    re.IGNORECASE,
)
GOOD_BALLOT_RE = re.compile(
    r"บัตรดี[^\n]*?([0-9๐-๙][0-9๐-๙,]*)",
    re.IGNORECASE,
)


@dataclass
class ValidationResult:
    doc_id: str
    expected_rows: int
    matched_rows: int
    missing_row_nums: list[int]
    missing_parties: list[str]
    expected_total_votes: int | None
    extracted_total_votes: int
    total_match: bool | None
    status: str


def extract_expected_total_votes(page1_markdown: str) -> int | None:
    text = page1_markdown.replace("\\.", ".")
    for pattern in (SECTION_41_RE, GOOD_BALLOT_RE):
        match = pattern.search(text)
        if not match:
            continue
        digits = extract_vote_digits(match.group(1))
        if digits:
            return int(digits)
    return None


def build_validation_result(
    *,
    doc_id: str,
    assigned_rows: list[dict[str, object]],
    expected_total_votes: int | None,
) -> ValidationResult:
    missing = [
        row for row in assigned_rows
        if not row["matched"] and str(row["party_name"]).strip()
    ]
    extracted_total_votes = sum(
        int(str(row["votes"]))
        for row in assigned_rows
        if (row["matched"] or not str(row["party_name"]).strip()) and str(row["votes"]).isdigit()
    )
    total_match = None if expected_total_votes is None else expected_total_votes == extracted_total_votes

    if missing:
        status = "needs_review_missing_rows"
    elif total_match is False:
        status = "needs_review_total_mismatch"
    elif total_match is None:
        status = "warning_missing_page1_total"
    else:
        status = "ok"

    return ValidationResult(
        doc_id=doc_id,
        expected_rows=len(assigned_rows),
        matched_rows=sum(
            1 for row in assigned_rows
            if row["matched"] or not str(row["party_name"]).strip()
        ),
        missing_row_nums=[int(row["row_num"]) for row in missing],
        missing_parties=[str(row["party_name"]) for row in missing],
        expected_total_votes=expected_total_votes,
        extracted_total_votes=extracted_total_votes,
        total_match=total_match,
        status=status,
    )
