from pathlib import Path, WindowsPath
from collections import defaultdict
import re

def list_png_files(root_path):
    return list(Path(root_path).glob("*.png"))

def filter_valid_files(files):
    return [f for f in files if f.is_file()]

def parse_file(file_path: WindowsPath):
    name = file_path.stem
    match = re.search(r'_page(\d+)$', name)
    if match:
        page = int(match.group(1))
        doc_id = name[:match.start()]
    else:
        page = 1
        doc_id = name
    return {"doc_id": doc_id, "page": page, "path": str(file_path)}

def map_parse(files):
    return [parse_file(f) for f in files]

def group_by_doc_id(parsed_files: list[dict]):
    groups = defaultdict(list)
    for file_info in parsed_files:
        groups[file_info["doc_id"]].append(file_info)
    return dict(groups)

def sort_pages(groups):
    return {doc_id: sorted(items, key=lambda x: x["page"]) for doc_id, items in groups.items()}

def to_documents(groups):
    return [
        {
            "doc_id": doc_id,
            "pages": [item["path"] for item in items],
            "num_pages": len(items)
        }
        for doc_id, items in groups.items()
    ]

def filter_party_list_only(documents):
    return [doc for doc in documents if doc["doc_id"].startswith("party_list")]

def filter_constituency_only(documents):
    return [doc for doc in documents if doc["doc_id"].startswith("constituency")]