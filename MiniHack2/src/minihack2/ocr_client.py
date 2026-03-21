from __future__ import annotations

from pathlib import Path
import os
import time


def _load_env_file() -> None:
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)


def _import_ocr_document():
    try:
        from typhoon_ocr import ocr_document
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `typhoon-ocr`. Install it with `pip install -r requirements.txt`."
        ) from exc
    return ocr_document


def ensure_api_key() -> None:
    _load_env_file()
    if os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return
    raise RuntimeError(
        "Set TYPHOON_OCR_API_KEY or OPENAI_API_KEY before calling Typhoon OCR."
    )


def ocr_image_to_markdown(
    image_path: Path,
    *,
    min_interval_seconds: float = 3.2,
    _last_call: list[float] | None = None,
) -> str:
    """OCR a single PNG/JPG image with basic client-side throttling.

    Typhoon documents currently list a limit of 20 requests per minute for `typhoon-ocr`,
    so the default interval stays a bit under that cap.
    """
    ensure_api_key()
    ocr_document = _import_ocr_document()

    if _last_call is None:
        _last_call = [0.0]
    elapsed = time.time() - _last_call[0]
    if elapsed < min_interval_seconds:
        time.sleep(min_interval_seconds - elapsed)

    markdown = ocr_document(
        pdf_or_image_path=str(image_path),
        api_key=os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )
    _last_call[0] = time.time()
    return markdown
