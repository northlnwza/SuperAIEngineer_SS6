# %% [markdown]
# # SuperAI Engineer OCR 2569
# Offline Colab pipeline using cached OCR output only.
#
# This notebook-style file is self-contained:
# - no imports from `src/minihack2`
# - no Typhoon OCR calls
# - uses cached markdown from `outputs/ocr_raw/`
# - rebuilds the internal submission, validation report, and Kaggle CSV
#
# Copy each `# %%` block into a separate Google Colab cell.


# %% [markdown]
# ## Cell 1: Mount Google Drive

from google.colab import drive

drive.mount("/content/drive")


# %% [markdown]
# ## Cell 2: Set The Data Folder
# Update only `DATA_DIR` to match your Google Drive path.
# The notebook will treat the parent folder of `data/` as the project root.

DATA_DIR = "/content/drive/MyDrive/MiniHack2/data"

import os
from pathlib import Path

os.chdir(str(Path(DATA_DIR).parent))
print("cwd =", os.getcwd())


# %% [markdown]
# ## Cell 3: Standard Library Imports

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
from difflib import SequenceMatcher
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unicodedata


# %% [markdown]
# ## Cell 4: Where Typhoon OCR Was Used
# This notebook rebuilds the final result from cached OCR markdown in `outputs/ocr_raw/`
# so that it does not need to call the Typhoon API again.
#
# Original OCR step used in the project:
#
# ```python
# from typhoon_ocr import ocr_document
#
# markdown = ocr_document(
#     pdf_or_image_path=str(image_path),
#     api_key=os.getenv("TYPHOON_OCR_API_KEY"),
# )
# ```
#
# After that OCR step, the pipeline:
# - parsed the markdown tables
# - aligned OCR rows to the template
# - validated totals
# - wrote the Kaggle submission


# %% [markdown]
# ## Cell 5: Optional Live Typhoon OCR Demo
# Leave this disabled by default to avoid API usage and rate-limit issues.

# from pathlib import Path
# import os
# from typhoon_ocr import ocr_document
#
# sample_image = Path(DATA_DIR).parent / "data" / "images" / "REPLACE_WITH_REAL_IMAGE.png"
# sample_markdown = ocr_document(
#     pdf_or_image_path=str(sample_image),
#     api_key=os.getenv("TYPHOON_OCR_API_KEY"),
# )
# print(sample_markdown[:3000])


# %% [markdown]
# ## Cell 6: Paths

DATA_DIR = Path(DATA_DIR)
ROOT = DATA_DIR.parent
IMAGES_DIR = DATA_DIR / "images"
TEMPLATE_PATH = DATA_DIR / "submission_template.csv"
CACHE_DIR = ROOT / "outputs" / "ocr_raw"
DEBUG_JSON_PATH = ROOT / "outputs" / "debug" / "parsed_rows_offline.json"
VALIDATION_JSON_PATH = ROOT / "outputs" / "debug" / "validation_report_offline.json"
BASELINE_OUTPUT_CSV = ROOT / "outputs" / "submissions" / "submission_typhoon_baseline_offline.csv"
KAGGLE_OUTPUT_CSV = ROOT / "outputs" / "submissions" / "submission_kaggle_offline.csv"

for path in [IMAGES_DIR, TEMPLATE_PATH, CACHE_DIR]:
    print(path, "exists=", path.exists())


# %% [markdown]
# ## Cell 7: All Pipeline Code In One Place

PAGE_SUFFIX_RE = re.compile(r"^(?P<doc_id>.+?)(?:_page(?P<page>\d+))?$")
TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
MD_ROW_RE = re.compile(r"^\|(.+)\|$")
NON_DIGIT_RE = re.compile(r"\D+")
PARENS_TEXT_RE = re.compile(r"\([^)]*\)")
PARENS_RE = re.compile(r"\(([^)]*)\)")
CLEAN_RE = re.compile(r"[^ก-๙]")
SECTION_41_RE = re.compile(r"(?:4|๔)\\?\.(?:1|๑)[^\n]*?([0-9๐-๙][0-9๐-๙,]*)", re.IGNORECASE)
GOOD_BALLOT_RE = re.compile(r"บัตรดี[^\n]*?([0-9๐-๙][0-9๐-๙,]*)", re.IGNORECASE)
PAGE_TOTAL_NUMBER_RE = re.compile(r"[0-9๐-๙][0-9๐-๙,\.]*")

THAI_TO_ARABIC_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
THAI_DIGIT_WORDS = {
    "ศูนย์": 0,
    "หนึ่ง": 1,
    "เอ็ด": 1,
    "สอง": 2,
    "ยี่": 2,
    "สาม": 3,
    "สี่": 4,
    "ห้า": 5,
    "หก": 6,
    "เจ็ด": 7,
    "แปด": 8,
    "เก้า": 9,
}
UNIT_VALUES = {
    "สิบ": 10,
    "ร้อย": 100,
    "พัน": 1000,
    "หมื่น": 10000,
    "แสน": 100000,
}


@dataclass(frozen=True)
class PageInfo:
    doc_id: str
    page_num: int
    path: Path


@dataclass
class OCRRow:
    row_num: int | None
    party_name: str
    votes: str
    vote_words_value: int | None
    raw_cells: list[str]
    source: str


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


class HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table":
            self._current_table = []
        elif tag == "tr" and self._current_table is not None:
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            self._current_row.append(normalize_whitespace("".join(self._current_cell)))
            self._current_cell = None
        elif tag == "tr" and self._current_table is not None and self._current_row is not None:
            if any(cell.strip() for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._current_table is not None:
            self.tables.append(self._current_table)
            self._current_table = None

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)


def parse_page_info(path: Path) -> PageInfo:
    match = PAGE_SUFFIX_RE.match(path.stem)
    if not match:
        raise ValueError(f"Unrecognized page name: {path.name}")
    page_str = match.group("page")
    page_num = int(page_str) if page_str else 1
    return PageInfo(doc_id=match.group("doc_id"), page_num=page_num, path=path)


def group_document_pages(images_dir: Path) -> dict[str, list[PageInfo]]:
    grouped: dict[str, list[PageInfo]] = {}
    for image_path in sorted(images_dir.glob("*.png")):
        info = parse_page_info(image_path)
        grouped.setdefault(info.doc_id, []).append(info)
    for pages in grouped.values():
        pages.sort(key=lambda item: item.page_num)
    return grouped


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_party_name(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = normalize_whitespace(text.strip())
    return text.replace(" ", "")


def normalize_digits(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(THAI_TO_ARABIC_DIGITS)
    return text


def extract_vote_digits(text: str) -> str:
    text = normalize_digits(text)
    text = PARENS_TEXT_RE.sub("", text)
    text = text.replace(",", "")
    digits = NON_DIGIT_RE.sub("", text)
    return digits


def extract_parenthetical_text(text: str) -> str | None:
    match = PARENS_RE.search(text)
    if not match:
        return None
    cleaned = CLEAN_RE.sub("", match.group(1))
    return cleaned or None


def thai_words_to_int(text: str) -> int | None:
    if not text:
        return None
    if "ล้าน" in text:
        parts = text.split("ล้าน")
        total = 0
        for idx, part in enumerate(parts):
            if idx < len(parts) - 1:
                left = thai_words_to_int(part)
                if left is None:
                    return None
                total = (total + left) * 1_000_000
            elif part:
                right = thai_words_to_int(part)
                if right is None:
                    return None
                total += right
        return total

    total = 0
    current = 0
    i = 0
    while i < len(text):
        matched = False
        for unit, value in UNIT_VALUES.items():
            if text.startswith(unit, i):
                multiplier = current if current != 0 else 1
                total += multiplier * value
                current = 0
                i += len(unit)
                matched = True
                break
        if matched:
            continue
        for word, value in THAI_DIGIT_WORDS.items():
            if text.startswith(word, i):
                current = value
                i += len(word)
                matched = True
                break
        if matched:
            continue
        return None
    return total + current


def extract_vote_words_value(text: str) -> int | None:
    words = extract_parenthetical_text(text)
    if not words:
        return None
    return thai_words_to_int(words)


def load_submission_template(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def write_submission(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_cached_markdown(page_path: Path, cache_dir: Path) -> str:
    cache_path = cache_dir / f"{page_path.stem}.md"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing OCR cache: {cache_path}")
    return cache_path.read_text(encoding="utf-8")


def to_row_num(text: str) -> int | None:
    digits = extract_vote_digits(text)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def cells_to_row(cells: list[str], source: str) -> OCRRow | None:
    if len(cells) < 3:
        return None

    row_num = to_row_num(cells[0])
    if len(cells) >= 4:
        cell1_digits = extract_vote_digits(cells[1])
        cell2_digits = extract_vote_digits(cells[2])
        cell3_digits = extract_vote_digits(cells[3])
        if not cell1_digits and cell2_digits and not cell3_digits:
            party_cell = cells[1]
            vote_cell = cells[2]
        else:
            party_cell = cells[2]
            vote_cell = cells[3]
    else:
        party_cell = cells[1]
        vote_cell = cells[2]

    party_name = normalize_whitespace(TAG_RE.sub("", party_cell))
    votes = extract_vote_digits(vote_cell)
    vote_words_value = extract_vote_words_value(vote_cell)
    if "รวมคะแนนทั้งสิ้น" in party_name:
        return None
    if not party_name and not votes:
        return None
    if row_num is None and not votes:
        return None
    return OCRRow(
        row_num=row_num,
        party_name=party_name,
        votes=votes,
        vote_words_value=vote_words_value,
        raw_cells=cells,
        source=source,
    )


def parse_html_tables(markdown: str, source: str) -> list[OCRRow]:
    parser = HTMLTableParser()
    for table_html in TABLE_RE.findall(markdown):
        parser.feed(table_html)
    rows: list[OCRRow] = []
    for table in parser.tables:
        for cells in table:
            candidate = cells_to_row(cells, source)
            if candidate is not None:
                rows.append(candidate)
    return rows


def parse_markdown_rows(markdown: str, source: str) -> list[OCRRow]:
    rows: list[OCRRow] = []
    for line in markdown.splitlines():
        match = MD_ROW_RE.match(line.strip())
        if not match:
            continue
        cells = [normalize_whitespace(cell) for cell in match.group(1).split("|")]
        candidate = cells_to_row(cells, source)
        if candidate is not None:
            rows.append(candidate)
    return rows


def parse_rows(markdown: str, source: str) -> list[OCRRow]:
    rows = parse_html_tables(markdown, source)
    if rows:
        return rows
    return parse_markdown_rows(markdown, source)


def choose_best_row(
    candidates: list[OCRRow],
    expected_row_num: int,
    expected_party_name: str,
    prefer_party_name: bool = False,
) -> OCRRow | None:
    expected_party_norm = normalize_party_name(expected_party_name)
    scored: list[tuple[int, int, OCRRow]] = []
    for row in candidates:
        row_match_penalty = 0 if row.row_num == expected_row_num else 1
        party_norm = normalize_party_name(row.party_name)
        if party_norm == expected_party_norm:
            party_penalty = 0
        elif expected_party_norm and expected_party_norm in party_norm:
            party_penalty = 1
        elif party_norm and party_norm in expected_party_norm:
            party_penalty = 1
        else:
            party_penalty = 2
        if prefer_party_name:
            scored.append((party_penalty, row_match_penalty, row))
        else:
            scored.append((row_match_penalty, party_penalty, row))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1], -len(item[2].votes)))
    best_primary, best_secondary, best_row = scored[0]
    if prefer_party_name and best_primary >= 2:
        return None
    if not prefer_party_name and best_primary >= 1 and best_secondary >= 2:
        return None
    return best_row


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


def build_validation_result(doc_id: str, assigned_rows: list[dict[str, object]], expected_total_votes: int | None) -> ValidationResult:
    missing = [row for row in assigned_rows if not row["matched"] and str(row["party_name"]).strip()]
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
        matched_rows=sum(1 for row in assigned_rows if row["matched"] or not str(row["party_name"]).strip()),
        missing_row_nums=[int(row["row_num"]) for row in missing],
        missing_parties=[str(row["party_name"]) for row in missing],
        expected_total_votes=expected_total_votes,
        extracted_total_votes=extracted_total_votes,
        total_match=total_match,
        status=status,
    )


def page_has_total_row(page_markdown: str, expected_total_votes: int | None) -> bool:
    if "รวมคะแนนทั้งสิ้น" not in page_markdown:
        return False
    if expected_total_votes is None:
        return True
    normalized = page_markdown.replace(".", "").replace(",", "")
    return str(expected_total_votes) in normalized or "รวมคะแนนทั้งสิ้น" in page_markdown


def extract_page_total_row_votes(page_markdown: str) -> int | None:
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


def fill_single_missing_from_total(assigned_rows: list[dict[str, object]], expected_total_votes: int | None) -> None:
    if expected_total_votes is None:
        return
    matched_rows = [row for row in assigned_rows if row["matched"] and str(row["votes"]).isdigit()]
    missing_rows = [row for row in assigned_rows if not row["matched"]]
    if len(missing_rows) != 1:
        return
    matched_sum = sum(int(str(row["votes"])) for row in matched_rows)
    residual = expected_total_votes - matched_sum
    if residual >= 0:
        missing_rows[0]["votes"] = str(residual)
        missing_rows[0]["matched"] = True
        missing_rows[0]["rescued"] = "filled_from_doc_total_residual"


def rescue_party_list_tail_rows(
    assigned_rows: list[dict[str, object]],
    expected_total_votes: int | None,
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
        suspicious_total_values = {value for value in (expected_total_votes, page_total_values.get(source)) if value is not None}
        is_suspicious_tail = int(row.get("row_num", 0)) >= 55 and vote_value > expected_total_votes * 0.5
        if page_total_flags.get(source, False) and (vote_value in suspicious_total_values or is_suspicious_tail):
            row["matched"] = False
            row["votes"] = "0"
            row["rescued"] = "dropped_total_like_tail_row"
    fill_single_missing_from_total(assigned_rows, expected_total_votes)


def assign_fuzzy_missing_constituency_rows(
    assigned_rows: list[dict[str, object]],
    parsed_rows: list[OCRRow],
    page_total_values: dict[str, int | None] | None = None,
) -> None:
    missing_rows = [row for row in assigned_rows if not row["matched"]]
    if not missing_rows:
        return
    used_parties = {str(row["party_name"]) for row in assigned_rows if row["matched"]}
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


def rescue_constituency_votes(
    assigned_rows: list[dict[str, object]],
    expected_total_votes: int | None,
    first_page_source_name: str | None = None,
) -> None:
    if expected_total_votes is None:
        return
    numeric_rows = [row for row in assigned_rows if row["matched"] and str(row["votes"]).isdigit()]
    if len(numeric_rows) >= 2:
        for row in numeric_rows:
            vote_value = int(str(row["votes"]))
            other_sum = sum(int(str(other["votes"])) for other in numeric_rows if other is not row)
            if vote_value == expected_total_votes and other_sum > 0:
                corrected = expected_total_votes - other_sum
                if 0 <= corrected < vote_value:
                    row["votes"] = str(corrected)
                    row["rescued"] = "rebalanced_from_doc_total"

    matched_rows = [row for row in assigned_rows if row["matched"] and str(row["votes"]).isdigit()]
    missing_rows = [row for row in assigned_rows if not row["matched"]]
    if len(missing_rows) == 1:
        matched_sum = sum(int(str(row["votes"])) for row in matched_rows)
        residual = expected_total_votes - matched_sum
        if residual >= 0:
            missing_rows[0]["votes"] = str(residual)
            missing_rows[0]["matched"] = True
            missing_rows[0]["rescued"] = "filled_from_doc_total_residual"

    matched_rows = [row for row in assigned_rows if row["matched"] and str(row["votes"]).isdigit()]
    if matched_rows and first_page_source_name:
        current_sum = sum(int(str(row["votes"])) for row in matched_rows)
        if current_sum != expected_total_votes:
            page1_rows = [row for row in matched_rows if row.get("matched_source") == first_page_source_name]
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


def rescue_with_word_values(assigned_rows: list[dict[str, object]], expected_total_votes: int | None) -> None:
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

    base_sum = sum(int(str(row["votes"])) for row in assigned_rows if row["matched"] and str(row["votes"]).isdigit())
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


def collect_doc_rows(doc_pages: list[PageInfo], cache_dir: Path) -> list[OCRRow]:
    rows: list[OCRRow] = []
    for page in doc_pages:
        markdown = read_cached_markdown(page.path, cache_dir)
        rows.extend(parse_rows(markdown, page.path.name))
    return rows


def build_submission_offline(
    images_dir: Path,
    template_path: Path,
    cache_dir: Path,
    output_csv: Path,
    debug_json_path: Path,
    validation_json_path: Path,
) -> None:
    grouped_pages = group_document_pages(images_dir)
    template_rows = load_submission_template(template_path)
    rows_by_doc: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in template_rows:
        rows_by_doc[row["doc_id"]].append(row)

    debug_payload: dict[str, object] = {}
    validation_payload: dict[str, object] = {}

    for doc_id, submission_rows in rows_by_doc.items():
        doc_pages = grouped_pages.get(doc_id, [])
        if not doc_pages:
            continue
        parsed_rows = collect_doc_rows(doc_pages, cache_dir)
        page1_markdown = read_cached_markdown(doc_pages[0].path, cache_dir)
        expected_total_votes = extract_expected_total_votes(page1_markdown)
        page_total_flags = {
            page.path.name: page_has_total_row(read_cached_markdown(page.path, cache_dir), expected_total_votes)
            for page in doc_pages
        }
        page_total_values = {
            page.path.name: extract_page_total_row_votes(read_cached_markdown(page.path, cache_dir))
            for page in doc_pages
        }

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
                    "vote_words_value": row.vote_words_value,
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
                and int(best.votes) in {value for value in (expected_total_votes, page_total_values.get(best.source)) if value is not None}
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
            assign_fuzzy_missing_constituency_rows(assigned_rows, parsed_rows, page_total_values)
            rescue_constituency_votes(assigned_rows, expected_total_votes, doc_pages[0].path.name)
            rescue_with_word_values(assigned_rows, expected_total_votes)
            rescue_constituency_votes(assigned_rows, expected_total_votes, doc_pages[0].path.name)
            votes_by_party = {str(row["party_name"]): str(row["votes"]) for row in assigned_rows if row["matched"]}
            for submission_row in submission_rows:
                if submission_row["party_name"] in votes_by_party:
                    submission_row["votes"] = votes_by_party[submission_row["party_name"]]
        else:
            rescue_party_list_tail_rows(assigned_rows, expected_total_votes, page_total_flags, page_total_values)
            fill_single_missing_from_total(assigned_rows, expected_total_votes)
            votes_by_row_num = {int(row["row_num"]): str(row["votes"]) for row in assigned_rows if row["matched"]}
            for submission_row in submission_rows:
                row_num = int(submission_row["row_num"])
                if row_num in votes_by_row_num:
                    submission_row["votes"] = votes_by_row_num[row_num]

        validation = build_validation_result(doc_id, assigned_rows, expected_total_votes)
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

    debug_json_path.parent.mkdir(parents=True, exist_ok=True)
    debug_json_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    validation_json_path.parent.mkdir(parents=True, exist_ok=True)
    validation_json_path.write_text(json.dumps(validation_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def convert_internal_to_kaggle(internal_csv: Path, kaggle_csv: Path) -> None:
    with internal_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    kaggle_csv.parent.mkdir(parents=True, exist_ok=True)
    with kaggle_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "votes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "votes": row["votes"]})


print("offline pipeline code loaded")


# %% [markdown]
# ## Cell 8: Inspect Cached OCR For One Example Document
# This shows input and output without calling Typhoon again.

grouped_pages = group_document_pages(IMAGES_DIR)
doc_id = "party_list_10_2"
doc_pages = grouped_pages[doc_id]

print("doc_id =", doc_id)
print("pages =", [page.path.name for page in doc_pages])

markdown_preview = read_cached_markdown(doc_pages[1].path, CACHE_DIR)
print(markdown_preview[:3000])


# %% [markdown]
# ## Cell 9: Parse One Cached OCR Page Into Structured Rows

rows = parse_rows(markdown_preview, doc_pages[1].path.name)
print("parsed_row_count =", len(rows))
for row in rows[:5]:
    print(
        {
            "row_num": row.row_num,
            "party_name": row.party_name,
            "votes": row.votes,
            "vote_words_value": row.vote_words_value,
            "source": row.source,
        }
    )


# %% [markdown]
# ## Cell 10: Build The Full Submission Offline From Cached OCR
# No Typhoon calls happen here.

build_submission_offline(
    images_dir=IMAGES_DIR,
    template_path=TEMPLATE_PATH,
    cache_dir=CACHE_DIR,
    output_csv=BASELINE_OUTPUT_CSV,
    debug_json_path=DEBUG_JSON_PATH,
    validation_json_path=VALIDATION_JSON_PATH,
)

convert_internal_to_kaggle(BASELINE_OUTPUT_CSV, KAGGLE_OUTPUT_CSV)

print("baseline =", BASELINE_OUTPUT_CSV)
print("kaggle =", KAGGLE_OUTPUT_CSV)


# %% [markdown]
# ## Cell 11: Show Validation Summary

validation_data = json.loads(VALIDATION_JSON_PATH.read_text(encoding="utf-8"))
status_counts = Counter(item["status"] for item in validation_data.values())
print("status_counts =", dict(status_counts))

flagged = [item for item in validation_data.values() if item["status"] != "ok"]
flagged.sort(key=lambda item: (item["status"], item["doc_id"]))
for item in flagged[:20]:
    print(
        item["doc_id"],
        item["status"],
        f"matched={item['matched_rows']}/{item['expected_rows']}",
        f"total={item['extracted_total_votes']}/{item['expected_total_votes']}",
        f"missing_rows={item['missing_row_nums']}",
    )


# %% [markdown]
# ## Cell 12: Inspect One Full Debug Record

debug_data = json.loads(DEBUG_JSON_PATH.read_text(encoding="utf-8"))
doc_debug = debug_data["party_list_10_2"]

print("expected_total_votes =", doc_debug["expected_total_votes"])
print("validation =", doc_debug["validation"])
print("first_assigned_rows =")
for row in doc_debug["assigned_rows"][:8]:
    print(row)


# %% [markdown]
# ## Cell 13: Preview Final Kaggle Submission

with KAGGLE_OUTPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    for idx, row in enumerate(reader):
        print(row)
        if idx >= 9:
            break


# %% [markdown]
# ## Cell 14: Optional Download

from google.colab import files

# Uncomment to download the final file.
# files.download(str(KAGGLE_OUTPUT_CSV))
