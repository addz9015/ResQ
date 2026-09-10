"""
Document ingestion  (src/document_ingest.py)  —  SIH 2026 PS26070.

Turns an uploaded file into plain text for the Official Warnings tab. It does
NOT parse a bulletin and it does NOT classify one — that is
src/imd_bulletin.py's job. This module only:

  * accepts .pdf and .txt, rejecting everything else by extension AND by the
    file's leading magic bytes,
  * caps the upload at 20 MB and a PDF at 200 pages,
  * pulls text out of a PDF page by page with pdfplumber, keeping the page
    number of every chunk so a multi-storm report can be reported by page range,
  * strips the header/footer lines that repeat on (almost) every page,
  * returns {text, page_count, per_page_text, chars_extracted}.

HARD RULES
  * Nothing from the file is ever executed.
  * The upload is processed in memory and discarded — never written to disk.
  * No OCR. A scanned / image-only PDF simply yields little or no text; the
    caller detects that (NO_TEXT_LAYER) and asks the user to paste instead.
"""
from __future__ import annotations

import io
import os
import re

MAX_BYTES = 20 * 1024 * 1024        # 20 MB
MAX_PAGES = 200
ALLOWED_EXT = {".pdf", ".txt"}

# magic bytes of formats we explicitly refuse even if the name ends in .pdf/.txt
_REJECT_SIGNATURES = {
    b"MZ": "Windows executable",
    b"\x7fELF": "ELF executable",
    b"PK\x03\x04": "ZIP / Office document",
    b"\xd0\xcf\x11\xe0": "legacy Office document",
    b"\x1f\x8b": "gzip archive",
    b"Rar!": "RAR archive",
    b"\xca\xfe\xba\xbe": "Mach-O / Java class",
}


class DocumentError(ValueError):
    """A file we will not process (wrong type, too big, corrupt, unsafe)."""


# ---------------------------------------------------------------------------
def _looks_like_text(data: bytes) -> bool:
    head = data[:8192]
    if b"\x00" in head:                       # NUL byte => binary
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError:
        # allow a stray non-UTF-8 byte only if the file is otherwise printable
        printable = sum(c == 9 or c == 10 or c == 13 or 32 <= c < 127
                        for c in head)
        return head and printable / len(head) > 0.85


def _sniff(data: bytes) -> str | None:
    for sig, _label in _REJECT_SIGNATURES.items():
        if data.startswith(sig):
            return None
    if data.startswith(b"%PDF-"):
        return "pdf"
    if _looks_like_text(data):
        return "txt"
    return None


# ---------------------------------------------------------------------------
_norm = lambda s: re.sub(r"\s+", " ", s or "").strip().lower()


def _strip_repeating_headers(pages: list[str]) -> list[str]:
    """Remove the first/last lines that recur on most pages (running header /
    footer, page numbers). Only applied when there are at least 3 pages."""
    if len(pages) < 3:
        return pages
    threshold = max(3, int(round(len(pages) * 0.6)))
    edge_counts: dict[str, int] = {}
    for pg in pages:
        lines = [ln for ln in pg.splitlines() if ln.strip()]
        edge = lines[:3] + lines[-3:]
        for ln in set(_norm(ln) for ln in edge):
            if ln:
                edge_counts[ln] = edge_counts.get(ln, 0) + 1
    repeating = {ln for ln, n in edge_counts.items() if n >= threshold}
    if not repeating:
        return pages

    def clean(pg: str) -> str:
        lines = pg.splitlines()
        # drop repeating lines only while they sit at the very top / bottom
        while lines and (not lines[0].strip()
                         or _norm(lines[0]) in repeating
                         or re.fullmatch(r"\s*(page\s*)?\d+\s*", lines[0].lower())):
            lines.pop(0)
        while lines and (not lines[-1].strip()
                         or _norm(lines[-1]) in repeating
                         or re.fullmatch(r"\s*(page\s*)?\d+\s*", lines[-1].lower())):
            lines.pop()
        return "\n".join(lines)

    return [clean(pg) for pg in pages]


# ---------------------------------------------------------------------------
def _extract_pdf(data: bytes) -> dict:
    import pdfplumber

    try:
        pdf = pdfplumber.open(io.BytesIO(data))
    except Exception as e:                                # noqa: BLE001
        raise DocumentError(f"could not open the PDF ({e.__class__.__name__})")
    try:
        n = len(pdf.pages)
        if n > MAX_PAGES:
            raise DocumentError(f"PDF has {n} pages; the limit is {MAX_PAGES}")
        per_page = []
        for page in pdf.pages:
            try:
                per_page.append(page.extract_text() or "")
            except Exception:                            # noqa: BLE001
                per_page.append("")
    finally:
        pdf.close()

    per_page = _strip_repeating_headers(per_page)
    text = "\n\n".join(per_page).strip()
    return {
        "text": text,
        "page_count": len(per_page),
        "per_page_text": per_page,
        "chars_extracted": len(re.sub(r"\s", "", text)),
    }


def _extract_txt(data: bytes) -> dict:
    text = data.decode("utf-8", errors="replace").strip()
    return {
        "text": text,
        "page_count": 1,
        "per_page_text": [text],
        "chars_extracted": len(re.sub(r"\s", "", text)),
    }


# ---------------------------------------------------------------------------
def extract_document(filename: str, data: bytes) -> dict:
    """Validate and extract. Raises DocumentError for anything we refuse.

    Returns {text, page_count, per_page_text, chars_extracted}. `page_count`
    is 1 for .txt; `per_page_text[i]` is the text of page i+1 for a PDF.
    """
    if not data:
        raise DocumentError("the uploaded file is empty")
    if len(data) > MAX_BYTES:
        raise DocumentError(
            f"file is {len(data) / 1e6:.1f} MB; the limit is {MAX_BYTES // (1024 * 1024)} MB")

    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise DocumentError("only .pdf and .txt files are accepted")

    kind = _sniff(data)
    for sig, label in _REJECT_SIGNATURES.items():
        if data.startswith(sig):
            raise DocumentError(f"file looks like a {label}, not a {ext} document")
    if kind is None:
        raise DocumentError(f"file content does not look like a {ext} document")
    if ext == ".pdf" and kind != "pdf":
        raise DocumentError("file does not start with a PDF header (%PDF-)")
    if ext == ".txt" and kind != "txt":
        raise DocumentError("file is not plain text")

    return _extract_pdf(data) if ext == ".pdf" else _extract_txt(data)
