from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minihack2.pipeline import build_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Kaggle submission using Typhoon OCR cached markdown output."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=ROOT / "data" / "images",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "data" / "submission_template.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "submissions" / "submission_typhoon_baseline.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "outputs" / "ocr_raw",
    )
    parser.add_argument(
        "--debug-json",
        type=Path,
        default=ROOT / "outputs" / "debug" / "parsed_rows.json",
    )
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=ROOT / "outputs" / "debug" / "validation_report.json",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        default=[],
        help="Restrict processing to one or more doc_ids. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Process only the first N documents from the submission template order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_submission(
        images_dir=args.images_dir,
        template_path=args.template,
        output_csv=args.output,
        cache_dir=args.cache_dir,
        debug_json_path=args.debug_json,
        validation_json_path=args.validation_json,
        doc_ids=set(args.doc_id) if args.doc_id else None,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()
