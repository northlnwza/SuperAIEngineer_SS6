from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import re
from typing import Iterable

from .normalize import extract_vote_digits, normalize_digits, normalize_party_name, normalize_whitespace
from .thai_number_words import extract_vote_words_value


TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
MD_ROW_RE = re.compile(r"^\|(.+)\|$")


@dataclass
class OCRRow:
    row_num: int | None
    party_name: str
    votes: str
    vote_words_value: int | None
    raw_cells: list[str]
    source: str


class _HTMLTableParser(HTMLParser):
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


def _to_row_num(text: str) -> int | None:
    digits = extract_vote_digits(text)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _cells_to_row(cells: list[str], source: str) -> OCRRow | None:
    if len(cells) < 3:
        return None

    row_num = _to_row_num(cells[0])
    if len(cells) >= 4:
        cell1_digits = extract_vote_digits(cells[1])
        cell2_digits = extract_vote_digits(cells[2])
        cell3_digits = extract_vote_digits(cells[3])

        # Party-list tables sometimes have 4 columns because the last column is just "หมายเหตุ".
        # In that shape the party is column 2 and the votes are column 3.
        if not cell1_digits and cell2_digits and not cell3_digits:
            party_cell = cells[1]
            vote_cell = cells[2]
        else:
            # Constituency tables use candidate name in column 2, party in column 3, votes in column 4.
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


def parse_html_tables(markdown: str, *, source: str) -> list[OCRRow]:
    parser = _HTMLTableParser()
    for table_html in TABLE_RE.findall(markdown):
        parser.feed(table_html)

    rows: list[OCRRow] = []
    for table in parser.tables:
        for cells in table:
            candidate = _cells_to_row(cells, source)
            if candidate is not None:
                rows.append(candidate)
    return rows


def parse_markdown_rows(markdown: str, *, source: str) -> list[OCRRow]:
    rows: list[OCRRow] = []
    for line in markdown.splitlines():
        match = MD_ROW_RE.match(line.strip())
        if not match:
            continue
        cells = [normalize_whitespace(cell) for cell in match.group(1).split("|")]
        candidate = _cells_to_row(cells, source)
        if candidate is not None:
            rows.append(candidate)
    return rows


def parse_rows(markdown: str, *, source: str) -> list[OCRRow]:
    rows = parse_html_tables(markdown, source=source)
    if rows:
        return rows
    return parse_markdown_rows(markdown, source=source)


def choose_best_row(
    candidates: Iterable[OCRRow],
    expected_row_num: int,
    expected_party_name: str,
    *,
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
