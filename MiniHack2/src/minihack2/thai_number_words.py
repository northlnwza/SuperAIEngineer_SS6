from __future__ import annotations

import re


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

PARENS_RE = re.compile(r"\(([^)]*)\)")
CLEAN_RE = re.compile(r"[^ก-๙]")


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
