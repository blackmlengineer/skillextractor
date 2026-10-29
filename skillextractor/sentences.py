"""Lightweight, dependency-free sentence splitting."""

from __future__ import annotations

import re
from typing import List

# Abbreviations that end in a period but do not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd",
    "co", "corp", "e.g", "i.e", "approx", "dept", "est", "no", "fig", "u.s", "u.k",
}

_BULLET_PREFIX = re.compile(r"^\s*(?:[-*•●▪]|\d+[.)])\s+")
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    """Split a paragraph (or several) into sentences.

    Line breaks and bullet points are treated as hard boundaries, which suits
    job descriptions. Within a line, sentences are split after ., ! or ?
    followed by whitespace and a capital letter or digit, except after common
    abbreviations and single-letter initials.
    """
    if not text or not text.strip():
        return []

    sentences: List[str] = []
    for line in re.split(r"[\r\n]+", text):
        line = _BULLET_PREFIX.sub("", line).strip()
        if not line:
            continue
        start = 0
        for m in _BOUNDARY.finditer(line):
            before = line[start : m.start()]
            if before.endswith("."):
                words = before[:-1].split()
                last = words[-1].lower().strip("\"'([") if words else ""
                if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
                    continue
            sentences.append(line[start : m.end()].strip())
            start = m.end()
        tail = line[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences
