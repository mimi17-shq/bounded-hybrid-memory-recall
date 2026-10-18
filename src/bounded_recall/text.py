from __future__ import annotations

import re


_CHANNEL = re.compile(r"<channel\b[^>]*>(.*?)</channel>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_CJK_RUN = re.compile(r"[\u3400-\u9fff]{2,}")
_ASCII_TERM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,}")
_QUOTED = re.compile(r"[“\"']([^”\"']{2,40})[”\"']")
_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)

_LOW_INFORMATION = {
    "嗯",
    "嗯嗯",
    "哦",
    "好的",
    "好呀",
    "收到",
    "知道了",
    "ok",
    "okay",
    "谢谢",
    "哈哈",
}


def normalize_query(raw: str) -> str:
    """Remove transport wrappers before retrieval sees the user's text."""
    raw = raw or ""
    matches = _CHANNEL.findall(raw)
    if matches:
        raw = "\n".join(matches)
    raw = _TAG.sub(" ", raw)
    return _SPACE.sub(" ", raw).strip()


def is_low_information(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered or lowered in _LOW_INFORMATION or _PUNCT_ONLY.fullmatch(lowered):
        return True
    meaningful = re.sub(r"[^A-Za-z0-9\u3400-\u9fff]", "", lowered)
    return len(meaningful) <= 1


def literal_terms(text: str, max_terms: int = 96) -> tuple[str, ...]:
    """Return CJK phrase windows plus quoted/ASCII terms, longest first."""
    candidates: set[str] = set()
    candidates.update(part.strip().lower() for part in _QUOTED.findall(text))
    candidates.update(part.lower() for part in _ASCII_TERM.findall(text))
    for run in _CJK_RUN.findall(text):
        upper = min(8, len(run))
        for size in range(2, upper + 1):
            for start in range(0, len(run) - size + 1):
                candidates.add(run[start : start + size])
    ordered = sorted((term for term in candidates if len(term) >= 2), key=lambda x: (-len(x), x))
    return tuple(ordered[:max_terms])


def longest_literal_match(query_terms: tuple[str, ...], searchable: str) -> tuple[int, str]:
    haystack = searchable.lower()
    for term in query_terms:
        if term in haystack:
            return len(term), term
    return 0, ""
