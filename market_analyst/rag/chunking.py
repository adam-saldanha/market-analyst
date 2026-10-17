"""Split a filing into overlapping chunks, tagged with their 10-K section."""
import re

from ..config import CHUNK_OVERLAP, CHUNK_SIZE
from ..schemas import Chunk

# Matches headings like "Item 1A. Risk Factors" or "ITEM 7."
SECTION_RE = re.compile(r"\bItem\s+\d+[A-C]?\.", re.IGNORECASE)

# Table-of-contents entries sit a few dozen characters apart; real section
# headings are separated by whole sections. A bigger gap marks the start of the body.
TOC_GAP = 1000


def _section_starts(text: str) -> list[tuple[int, str]]:
    """(position, heading) for each real section heading, skipping the table of contents."""
    matches = [(m.start(), m.group().strip()) for m in SECTION_RE.finditer(text)]
    for i in range(1, len(matches)):
        if matches[i][0] - matches[i - 1][0] > TOC_GAP:
            return matches[i:]
    return matches


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Slide a window of `size` chars forward by (size - overlap), ending each
    chunk at a sentence boundary where possible and tagging it with its 10-K section."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    sections = _section_starts(text)

    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # back up to the last sentence end, but not past the midpoint
            cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        section = None
        for pos, name in sections:
            if pos > start:
                break
            section = name
        chunks.append(Chunk(id=len(chunks), text=text[start:end].strip(), section=section))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)   # always advance
    return chunks
