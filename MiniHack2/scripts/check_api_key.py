from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minihack2.ocr_client import ensure_api_key


def main() -> None:
    try:
        ensure_api_key()
    except Exception as exc:
        print(f"missing: {exc}")
        raise SystemExit(1)
    print("ok: api key detected")


if __name__ == "__main__":
    main()
