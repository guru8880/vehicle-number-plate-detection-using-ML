"""Helpers for cleaning OCR text and validating Singapore plate patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass


PLATE_BODY_RE = re.compile(r"^[A-Z]{1,3}\d{1,4}[A-Z]$")
FORBIDDEN_PREFIX_LETTERS = {"I", "O"}
FORBIDDEN_SUFFIX_LETTERS = {"F", "I", "N", "O", "Q", "V", "W"}
SG_COMMON_PREFIX_STARTS = {"S", "E", "F", "G"}

ALPHA_TO_DIGIT = {
    "B": "8",
    "D": "0",
    "G": "6",
    "I": "1",
    "L": "1",
    "O": "0",
    "Q": "0",
    "S": "5",
    "Z": "2",
}

DIGIT_TO_ALPHA = {
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}

CHECKSUM_WEIGHTS = (9, 4, 5, 4, 3, 2)
CHECKSUM_LETTERS = "AZYXUTSRPMLKJHGEDCB"


@dataclass(frozen=True)
class NormalizedPlate:
    raw_text: str
    cleaned_text: str
    compact: str
    pretty: str
    prefix: str
    digits: str
    suffix: str
    valid_shape: bool
    checksum_expected: str
    checksum_match: bool
    format_score: float
    conversion_count: int


def clean_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def format_plate_for_display(plate: str) -> str:
    return clean_plate_text(plate)


def checksum_corrected_plate(normalized: NormalizedPlate) -> str:
    if not normalized.valid_shape:
        return ""
    return f"{normalized.prefix}{normalized.digits}{normalized.checksum_expected}"


def compute_checksum_letter(prefix: str, digits: str) -> str:
    digits = digits.zfill(4)

    if len(prefix) == 1:
        letter_values = [0, _letter_value(prefix[0])]
    elif len(prefix) == 2:
        letter_values = [_letter_value(prefix[0]), _letter_value(prefix[1])]
    else:
        letter_values = [_letter_value(prefix[-2]), _letter_value(prefix[-1])]

    values = letter_values + [int(char) for char in digits]
    checksum_value = sum(value * weight for value, weight in zip(values, CHECKSUM_WEIGHTS))
    return CHECKSUM_LETTERS[checksum_value % 19]


def normalize_singapore_plate(raw_text: str) -> NormalizedPlate:
    cleaned = clean_plate_text(raw_text)

    empty_plate = NormalizedPlate(
        raw_text=raw_text or "",
        cleaned_text=cleaned,
        compact="",
        pretty="",
        prefix="",
        digits="",
        suffix="",
        valid_shape=False,
        checksum_expected="",
        checksum_match=False,
        format_score=0.0,
        conversion_count=99,
    )

    if len(cleaned) < 3 or len(cleaned) > 8:
        return empty_plate

    candidates: list[NormalizedPlate] = []

    for prefix_len in range(1, 4):
        for digit_len in range(1, 5):
            if prefix_len + digit_len + 1 != len(cleaned):
                continue

            prefix_raw = cleaned[:prefix_len]
            digits_raw = cleaned[prefix_len : prefix_len + digit_len]
            suffix_raw = cleaned[-1]

            prefix, prefix_conversions = _convert_prefix(prefix_raw)
            digits, digit_conversions = _convert_digits(digits_raw)
            suffix, suffix_conversions = _convert_suffix(suffix_raw)

            if not prefix or not digits or not suffix:
                continue

            compact = f"{prefix}{digits}{suffix}"
            if not PLATE_BODY_RE.match(compact):
                continue

            if any(char in FORBIDDEN_PREFIX_LETTERS for char in prefix):
                continue

            if suffix in FORBIDDEN_SUFFIX_LETTERS:
                continue

            checksum_expected = compute_checksum_letter(prefix, digits)
            checksum_match = suffix == checksum_expected
            conversion_count = prefix_conversions + digit_conversions + suffix_conversions

            format_score = 0.52
            if prefix[0] in SG_COMMON_PREFIX_STARTS:
                format_score += 0.08
            if len(prefix) == 3:
                format_score += 0.04
            if len(digits) >= 3:
                format_score += 0.04
            if checksum_match:
                format_score += 0.27
            format_score -= min(conversion_count * 0.08, 0.24)
            format_score = max(0.0, min(1.0, format_score))

            candidates.append(
                NormalizedPlate(
                    raw_text=raw_text or "",
                    cleaned_text=cleaned,
                    compact=compact,
                    pretty=compact,
                    prefix=prefix,
                    digits=digits,
                    suffix=suffix,
                    valid_shape=True,
                    checksum_expected=checksum_expected,
                    checksum_match=checksum_match,
                    format_score=format_score,
                    conversion_count=conversion_count,
                )
            )

    if not candidates:
        return empty_plate

    candidates.sort(
        key=lambda item: (
            item.checksum_match,
            item.format_score,
            -item.conversion_count,
            item.prefix[0] in SG_COMMON_PREFIX_STARTS,
            len(item.prefix),
            len(item.digits),
        ),
        reverse=True,
    )
    return candidates[0]


def status_from_score(score: float, checksum_match: bool, variant_hits: int) -> tuple[str, str]:
    if checksum_match and score >= 0.82 and variant_hits >= 2:
        return "confirmed", "Confirmed"
    if score >= 0.68:
        return "review", "Needs Review"
    return "rejected", "Rejected"


def _convert_prefix(text: str) -> tuple[str, int]:
    converted = []
    conversions = 0
    for char in text:
        if char.isalpha():
            converted.append(char)
            continue
        mapped = DIGIT_TO_ALPHA.get(char)
        if not mapped:
            return "", 99
        converted.append(mapped)
        conversions += 1
    return "".join(converted), conversions


def _convert_digits(text: str) -> tuple[str, int]:
    converted = []
    conversions = 0
    for char in text:
        if char.isdigit():
            converted.append(char)
            continue
        mapped = ALPHA_TO_DIGIT.get(char)
        if not mapped:
            return "", 99
        converted.append(mapped)
        conversions += 1
    return "".join(converted), conversions


def _convert_suffix(char: str) -> tuple[str, int]:
    if char.isalpha():
        return char, 0
    mapped = DIGIT_TO_ALPHA.get(char)
    if not mapped:
        return "", 99
    return mapped, 1


def _letter_value(char: str) -> int:
    return ord(char) - 64
