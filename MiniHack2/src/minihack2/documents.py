from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


PAGE_SUFFIX_RE = re.compile(r"^(?P<doc_id>.+?)(?:_page(?P<page>\d+))?$")


@dataclass(frozen=True)
class PageInfo:
    doc_id: str
    page_num: int
    path: Path


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
