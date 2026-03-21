from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minihack2.documents import group_document_pages


def main() -> None:
    grouped = group_document_pages(ROOT / "data" / "images")
    type_counts = Counter(doc_id.split("_")[0] for doc_id in grouped)
    page_counts = Counter(len(pages) for pages in grouped.values())
    print(f"documents={len(grouped)}")
    print(f"types={dict(type_counts)}")
    print(f"pages_per_doc={dict(sorted(page_counts.items()))}")


if __name__ == "__main__":
    main()
