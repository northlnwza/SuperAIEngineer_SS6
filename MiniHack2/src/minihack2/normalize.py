from __future__ import annotations

import re
import unicodedata


THAI_TO_ARABIC_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
NON_DIGIT_RE = re.compile(r"\D+")
PARENS_TEXT_RE = re.compile(r"\([^)]*\)")


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
