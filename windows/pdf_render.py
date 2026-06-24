"""PDF page renderer using PyMuPDF (fitz).

Page indexing note: the LanceDB index stores page numbers as 1-indexed integers
(the OCR path uses page_index+1; native text path uses fitz's 0-indexed page+1).
So index `page` = 1 means the FIRST page; fitz page index = page - 1.

resolve_pdf(source_name) -> Path | None:
    Find a PDF in app_paths.library_dir() matching source_name robustly:
    - exact filename match
    - with/without ".pdf" extension
    - case-insensitive
    - ignoring minor punctuation differences (collapse non-alphanumeric chars for comparison)
    Returns Path or None if not found.

render_page(pdf_path, page_number, dpi=200) -> QImage:
    Render the given 1-indexed page_number from pdf_path at the given dpi.
    Uses fitz (PyMuPDF) page.get_pixmap(dpi=dpi) then converts to QImage.
    page_number is 1-indexed (as stored in index); internally uses page_number-1 for fitz.
    Returns a QImage (RGB32 or ARGB32 format).
    Raises ValueError if page_number is out of range.
    Keep fitz import local to avoid import-time side effects.
    Headless-safe (QImage works under offscreen).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QImage

from windows.app_paths import library_dir


def _normalize(s: str) -> str:
    """Lowercase and collapse non-alphanumeric chars to single spaces."""
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return s.strip()


def resolve_pdf(source_name: str) -> Optional[Path]:
    """Find a PDF in library_dir() matching source_name robustly.

    Tries in order:
    1. Exact filename match
    2. With/without .pdf extension (case-insensitive)
    3. Normalized comparison against all PDFs in library_dir()

    Returns Path or None if not found.
    """
    lib = library_dir()
    if not lib.exists():
        return None

    # Candidates: all PDF files
    pdfs = list(lib.glob("*.pdf")) + list(lib.glob("*.PDF"))

    # 1. Exact filename match
    for p in pdfs:
        if p.name == source_name:
            return p

    # 2. Case-insensitive, with/without .pdf
    name_lower = source_name.lower()
    name_no_ext = name_lower.removesuffix(".pdf")
    for p in pdfs:
        candidate = p.name.lower()
        candidate_no_ext = candidate.removesuffix(".pdf")
        if candidate == name_lower or candidate_no_ext == name_no_ext:
            return p

    # 3. Normalized comparison
    norm_source = _normalize(name_no_ext)
    for p in pdfs:
        norm_cand = _normalize(p.stem.lower())
        if norm_cand == norm_source:
            return p

    return None


def render_page(pdf_path: Path, page_number: int, dpi: int = 200) -> QImage:
    """Render a 1-indexed page from a PDF and return a QImage.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-indexed page number (as stored in LanceDB index).
        dpi: Resolution for rendering.

    Returns:
        QImage in RGB888 or RGBA8888 format.

    Raises:
        ValueError: If page_number is out of range.
    """
    import fitz  # local import to avoid import-time side effects

    doc = fitz.open(str(pdf_path))
    try:
        total = doc.page_count
        if page_number < 1 or page_number > total:
            raise ValueError(
                f"page_number {page_number} out of range (1–{total}) for {pdf_path.name}"
            )

        page = doc[page_number - 1]
        pixmap = page.get_pixmap(dpi=dpi)

        # Copy samples bytes to avoid dangling pointer after doc/pixmap is freed
        samples = bytes(pixmap.samples)
        w, h, stride, n = pixmap.width, pixmap.height, pixmap.stride, pixmap.n

        if n == 3:
            fmt = QImage.Format.Format_RGB888
        elif n == 4:
            fmt = QImage.Format.Format_RGBA8888
        else:
            # Fallback: convert to RGB via fitz
            pixmap = pixmap.tobytes("png")
            img = QImage()
            img.loadFromData(pixmap, "PNG")
            return img

        img = QImage(samples, w, h, stride, fmt)
        # .copy() detaches from the samples buffer
        return img.copy()
    finally:
        doc.close()
