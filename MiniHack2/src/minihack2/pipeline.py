from __future__ import annotations

from collections import defaultdict
import csv
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any

from .documents import group_document_pages
from .normalize import extract_vote_digits
from .ocr_client import ocr_image_to_markdown
from .parse_tables import OCRRow, choose_best_row, parse_rows
from .validate import build_validation_result, extract_expected_total_votes


PAGE_TOTAL_NUMBER_RE = re.compile(r"[0-9๐-๙][0-9๐-๙,\.]*")


def load_submission_template(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def write_submission(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cache_ocr_page(page_path: Path, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{page_path.stem}.md"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    markdown = ocr_image_to_markdown(page_path)
    cache_path.write_text(markdown, encoding="utf-8")
    return markdown


def collect_doc_rows(doc_id: str, doc_pages, cache_dir: Path) -> list[OCRRow]:
    rows: list[OCRRow] = []
    for page in doc_pages:
        markdown = cache_ocr_page(page.path, cache_dir)
        rows.extend(parse_rows(markdown, source=page.path.name))
    return rows


def _page_has_total_row(page_markdown: str, expected_total_votes: int | None) -> bool:
    if "รวมคะแนนทั้งสิ้น" not in page_markdown:
        return False
    if expected_total_votes is None:
        return True
    normalized = page_markdown.replace(".", "").replace(",", "")
    return str(expected_total_votes) in normalized or "รวมคะแนนทั้งสิ้น" in page_markdown


def _extract_page_total_row_votes(page_markdown: str) -> int | None:
    if "รวมคะแนนทั้งสิ้น" not in page_markdown:
        return None
    tail = page_markdown.split("รวมคะแนนทั้งสิ้น", 1)[1]
    match = PAGE_TOTAL_NUMBER_RE.search(tail[:50])
    if not match:
        return None
    digits = extract_vote_digits(match.group(0))
    if not digits:
        return None
    return int(digits)


def _rescue_constituency_votes(
    assigned_rows: list[dict[str, object]],
    expected_total_votes: int | None,
    *,
    first_page_source_name: str | None = None,
) -> None:
    if expected_total_votes is None:
        return

    numeric_rows = [
        row for row in assigned_rows
        if row["matched"] and str(row["votes"]).isdigit()
    ]
    if len(numeric_rows) < 2:
        return

    for row in numeric_rows:
        vote_value = int(str(row["votes"]))
        other_sum = sum(
            int(str(other["votes"]))
            for other in numeric_rows
            if other is not row
        )
        # Common OCR failure: one candidate row is mistaken as the overall 4.1 total.
        if vote_value == expected_total_votes and other_sum > 0:
            corrected = expected_total_votes - other_sum
            if 0 <= corrected < vote_value:
                row["votes"] = str(corrected)
                row["rescued"] = "rebalanced_from_doc_total"

    matched_rows = [
        row for row in assigned_rows
        if row["matched"] and str(row["votes"]).isdigit()
    ]
    missing_rows = [row for row in assigned_rows if not row["matched"]]
    if len(missing_rows) == 1:
        matched_sum = sum(int(str(row["votes"])) for row in matched_rows)
        residual = expected_total_votes - matched_sum
        if residual >= 0:
            missing_rows[0]["votes"] = str(residual)
            missing_rows[0]["matched"] = True
            missing_rows[0]["rescued"] = "filled_from_doc_total_residual"

    matched_rows = [
        row for row in assigned_rows
        if row["matched"] and str(row["votes"]).isdigit()
    ]
    if not matched_rows:
        return

    current_sum = sum(int(str(row["votes"])) for row in matched_rows)
    if current_sum != expected_total_votes and first_page_source_name:
        page1_rows = [
            row for row in matched_rows
            if row.get("matched_source") == first_page_source_name
        ]
        if len(page1_rows) == 1:
            other_sum = sum(
                int(str(row["votes"]))
                for row in matched_rows
                if row.get("matched_source") != first_page_source_name
            )
            residual = expected_total_votes - other_sum
            if residual >= 0:
                page1_rows[0]["votes"] = str(residual)
                page1_rows[0]["rescued"] = "rebalanced_single_page1_row"


def _fill_single_missing_from_total(
    assigned_rows: list[dict[str, object]],
    expected_total_votes: int | None,
) -> None:
    if expected_total_votes is None:
        return
    matched_rows = [
        row for row in assigned_rows
        if row["matched"] and str(row["votes"]).isdigit()
    ]
    missing_rows = [row for row in assigned_rows if not row["matched"]]
    if len(missing_rows) != 1:
        return
    matched_sum = sum(int(str(row["votes"])) for row in matched_rows)
    residual = expected_total_votes - matched_sum
    if residual >= 0:
        missing_rows[0]["votes"] = str(residual)
        missing_rows[0]["matched"] = True
        missing_rows[0]["rescued"] = "filled_from_doc_total_residual"


def _rescue_party_list_tail_rows(
    assigned_rows: list[dict[str, object]],
    expected_total_votes: int | None,
    *,
    page_total_flags: dict[str, bool],
    page_total_values: dict[str, int | None],
) -> None:
    if expected_total_votes is None:
        return

    for row in assigned_rows:
        if not row["matched"] or not str(row["votes"]).isdigit():
            continue
        source = str(row.get("matched_source") or "")
        vote_value = int(str(row["votes"]))
        suspicious_total_values = {
            value for value in (expected_total_votes, page_total_values.get(source)) if value is not None
        }
        is_suspicious_tail = (
            int(row.get("row_num", 0)) >= 55
            and vote_value > expected_total_votes * 0.5
        )
        if page_total_flags.get(source, False) and (
            vote_value in suspicious_total_values or is_suspicious_tail
        ):
            row["matched"] = False
            row["votes"] = "0"
            row["rescued"] = "dropped_total_like_tail_row"

    _fill_single_missing_from_total(assigned_rows, expected_total_votes)


def _assign_fuzzy_missing_constituency_rows(
    assigned_rows: list[dict[str, object]],
    parsed_rows: list[OCRRow],
    *,
    page_total_values: dict[str, int | None] | None = None,
) -> None:
    missing_rows = [row for row in assigned_rows if not row["matched"]]
    if not missing_rows:
        return

    used_parties = {
        str(row["party_name"])
        for row in assigned_rows
        if row["matched"]
    }
    candidate_rows = [
        row for row in parsed_rows
        if (
            row.party_name
            and row.party_name not in used_parties
            and row.votes
            and (
                page_total_values is None
                or page_total_values.get(row.source) is None
                or int(row.votes) != page_total_values.get(row.source)
            )
        )
    ]

    for missing in missing_rows:
        best_score = 0.0
        best_row: OCRRow | None = None
        for candidate in candidate_rows:
            score = SequenceMatcher(None, missing["party_name"], candidate.party_name).ratio()
            if score > best_score:
                best_score = score
                best_row = candidate
        if best_row is not None and best_score >= 0.7:
            missing["votes"] = best_row.votes
            missing["matched"] = True
            missing["matched_source"] = best_row.source
            missing["rescued"] = f"fuzzy_party_match:{best_row.party_name}:{best_score:.3f}"
            candidate_rows.remove(best_row)


def _rescue_with_word_values(
    assigned_rows: list[dict[str, object]],
    expected_total_votes: int | None,
) -> None:
    if expected_total_votes is None:
        return

    candidates = [
        row for row in assigned_rows
        if (
            row["matched"]
            and str(row["votes"]).isdigit()
            and isinstance(row.get("vote_words_value"), int)
            and int(row["votes"]) != int(row["vote_words_value"])
        )
    ]
    if not candidates or len(candidates) > 12:
        return

    base_sum = sum(
        int(str(row["votes"]))
        for row in assigned_rows
        if row["matched"] and str(row["votes"]).isdigit()
    )
    best_diff = abs(base_sum - expected_total_votes)
    best_mask = 0

    for mask in range(1, 1 << len(candidates)):
        trial_sum = base_sum
        for idx, row in enumerate(candidates):
            if mask & (1 << idx):
                trial_sum -= int(str(row["votes"]))
                trial_sum += int(row["vote_words_value"])
        diff = abs(trial_sum - expected_total_votes)
        if diff < best_diff:
            best_diff = diff
            best_mask = mask
            if diff == 0:
                break

    if best_mask == 0:
        return

    for idx, row in enumerate(candidates):
        if best_mask & (1 << idx):
            row["votes"] = str(int(row["vote_words_value"]))
            row["rescued"] = f"word_value_substitution:{row['vote_words_value']}"


def build_submission(
    *,
    images_dir: Path,
    template_path: Path,
    output_csv: Path,
    cache_dir: Path,
    debug_json_path: Path | None = None,
    validation_json_path: Path | None = None,
    doc_ids: set[str] | None = None,
    max_docs: int | None = None,
) -> None:
    grouped_pages = group_document_pages(images_dir)
    template_rows = load_submission_template(template_path)
    rows_by_doc: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in template_rows:
        rows_by_doc[row["doc_id"]].append(row)

    debug_payload: dict[str, Any] = {}
    validation_payload: dict[str, Any] = {}
    processed_docs = 0

    for doc_id, submission_rows in rows_by_doc.items():
        if doc_ids is not None and doc_id not in doc_ids:
            continue
        if max_docs is not None and processed_docs >= max_docs:
            break
        doc_pages = grouped_pages.get(doc_id, [])
        if not doc_pages:
            continue
        parsed_rows = collect_doc_rows(doc_id, doc_pages, cache_dir)
        page1_markdown = cache_ocr_page(doc_pages[0].path, cache_dir)
        expected_total_votes = extract_expected_total_votes(page1_markdown)
        page_total_flags = {
            page.path.name: _page_has_total_row(
                cache_ocr_page(page.path, cache_dir),
                expected_total_votes,
            )
            for page in doc_pages
        }
        page_total_values = {
            page.path.name: _extract_page_total_row_votes(cache_ocr_page(page.path, cache_dir))
            for page in doc_pages
        }
        processed_docs += 1
        debug_payload[doc_id] = {
            "pages": [str(page.path) for page in doc_pages],
            "expected_total_votes": expected_total_votes,
            "page_total_flags": page_total_flags,
            "page_total_values": page_total_values,
            "parsed_rows": [
                {
                    "row_num": row.row_num,
                    "party_name": row.party_name,
                    "votes": row.votes,
                    "source": row.source,
                    "raw_cells": row.raw_cells,
                }
                for row in parsed_rows
            ],
        }

        assigned_rows: list[dict[str, object]] = []
        prefer_party_name = doc_id.startswith("constituency_")
        for submission_row in submission_rows:
            best = choose_best_row(
                parsed_rows,
                expected_row_num=int(submission_row["row_num"]),
                expected_party_name=submission_row["party_name"],
                prefer_party_name=prefer_party_name,
            )
            matched = bool(best and best.votes)
            if (
                matched
                and expected_total_votes is not None
                and best is not None
                and best.votes.isdigit()
                and page_total_flags.get(best.source, False)
                and int(best.votes)
                in {
                    value
                    for value in (
                        expected_total_votes,
                        page_total_values.get(best.source),
                    )
                    if value is not None
                }
            ):
                matched = False
            if best and best.votes:
                submission_row["votes"] = best.votes if matched else "0"
            assigned_rows.append(
                {
                    "row_num": int(submission_row["row_num"]),
                    "party_name": submission_row["party_name"],
                    "votes": submission_row["votes"],
                    "vote_words_value": best.vote_words_value if best else None,
                    "matched": matched,
                    "matched_source": best.source if best else None,
                }
            )

        if prefer_party_name:
            _assign_fuzzy_missing_constituency_rows(
                assigned_rows,
                parsed_rows,
                page_total_values=page_total_values,
            )
            _rescue_constituency_votes(
                assigned_rows,
                expected_total_votes,
                first_page_source_name=doc_pages[0].path.name,
            )
            _rescue_with_word_values(assigned_rows, expected_total_votes)
            _rescue_constituency_votes(
                assigned_rows,
                expected_total_votes,
                first_page_source_name=doc_pages[0].path.name,
            )
            votes_by_party = {
                str(row["party_name"]): str(row["votes"])
                for row in assigned_rows
                if row["matched"]
            }
            for submission_row in submission_rows:
                party_name = submission_row["party_name"]
                if party_name in votes_by_party:
                    submission_row["votes"] = votes_by_party[party_name]
        else:
            _rescue_party_list_tail_rows(
                assigned_rows,
                expected_total_votes,
                page_total_flags=page_total_flags,
                page_total_values=page_total_values,
            )
            _fill_single_missing_from_total(assigned_rows, expected_total_votes)
            votes_by_row_num = {
                int(row["row_num"]): str(row["votes"])
                for row in assigned_rows
                if row["matched"]
            }
            for submission_row in submission_rows:
                row_num = int(submission_row["row_num"])
                if row_num in votes_by_row_num:
                    submission_row["votes"] = votes_by_row_num[row_num]

        validation = build_validation_result(
            doc_id=doc_id,
            assigned_rows=assigned_rows,
            expected_total_votes=expected_total_votes,
        )
        validation_payload[doc_id] = {
            "doc_id": validation.doc_id,
            "expected_rows": validation.expected_rows,
            "matched_rows": validation.matched_rows,
            "missing_row_nums": validation.missing_row_nums,
            "missing_parties": validation.missing_parties,
            "expected_total_votes": validation.expected_total_votes,
            "extracted_total_votes": validation.extracted_total_votes,
            "total_match": validation.total_match,
            "status": validation.status,
        }
        debug_payload[doc_id]["assigned_rows"] = assigned_rows
        debug_payload[doc_id]["validation"] = validation_payload[doc_id]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_submission(output_csv, template_rows)

    if debug_json_path is not None:
        debug_json_path.parent.mkdir(parents=True, exist_ok=True)
        debug_json_path.write_text(
            json.dumps(debug_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if validation_json_path is not None:
        validation_json_path.parent.mkdir(parents=True, exist_ok=True)
        validation_json_path.write_text(
            json.dumps(validation_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
