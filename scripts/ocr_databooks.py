"""
OCR pipeline for TrueBeam Field Service Databooks.

Run with:
    PYTHONPATH=E:\\engineer-companion python scripts/ocr_databooks.py --doc both

Or import and call run_ocr() directly.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import structlog

from core.indexer import is_low_quality_chunk

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy-initialised OCR engines (avoid slow PaddlePaddle import at module load)
# ---------------------------------------------------------------------------
_pp_structure = None
_paddle_ocr = None


def _get_pp_structure():
    global _pp_structure
    if _pp_structure is None:
        from paddleocr import PPStructure
        _pp_structure = PPStructure(table=True, ocr=True, use_gpu=False, show_log=False)
    return _pp_structure


def _get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(lang='en', use_gpu=False, show_log=False, enable_mkldnn=False)
    return _paddle_ocr


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _render_page(page, dpi: int = 300) -> np.ndarray:
    """Render a fitz page to a BGR numpy array at the given DPI."""
    import fitz
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    # PyMuPDF returns RGB; convert to BGR for OpenCV/PaddleOCR
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _clahe_enhance(bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement only (no binarization, no deskew)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# HTML table → markdown
# ---------------------------------------------------------------------------

def _html_table_to_markdown(html: str) -> str:
    """Convert a simple HTML table string to a pipe-delimited markdown table."""
    for tag in ('<html>', '</html>', '<body>', '</body>',
                '<table>', '</table>', '<thead>', '</thead>',
                '<tbody>', '</tbody>'):
        html = html.replace(tag, '')

    rows: List[List[str]] = []
    for tr_match in re.finditer(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE):
        tr_content = tr_match.group(1)
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', tr_content, re.DOTALL | re.IGNORECASE)
        clean_cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        rows.append(clean_cells)

    if not rows:
        return ''

    lines: List[str] = []
    for i, row in enumerate(rows):
        lines.append('| ' + ' | '.join(row) + ' |')
        if i == 0:
            lines.append('| ' + ' | '.join(['---'] * len(row)) + ' |')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# OCR pipeline for a single page image
# ---------------------------------------------------------------------------

def _ocr_image(img_bgr: np.ndarray) -> str:
    """Run PP-Structure then fall back to basic PaddleOCR for non-table regions."""
    structure = _get_pp_structure()

    try:
        regions = structure(img_bgr)
    except Exception as exc:
        log.warning("ppstructure_failed", error=str(exc))
        regions = []

    if regions:
        text_parts: List[str] = []
        has_non_table = False

        for region in regions:
            region_type = region.get('type', '').lower()
            res = region.get('res', {})

            if region_type == 'table':
                html = res.get('html', '') if isinstance(res, dict) else ''
                if html:
                    text_parts.append(_html_table_to_markdown(html))
                else:
                    if isinstance(res, list):
                        for item in res:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                txt = item[1][0] if isinstance(item[1], (list, tuple)) else str(item[1])
                                text_parts.append(txt)
            else:
                has_non_table = True

        if has_non_table:
            fallback = _ocr_fallback(img_bgr)
            if fallback:
                text_parts.append(fallback)

        return '\n\n'.join(p for p in text_parts if p.strip())

    return _ocr_fallback(img_bgr)


def _ocr_fallback(img_bgr: np.ndarray) -> str:
    """Basic PaddleOCR with reading-order reconstruction."""
    ocr = _get_paddle_ocr()

    try:
        result = ocr.ocr(img_bgr, cls=True)
    except Exception as exc:
        log.warning("paddleocr_failed", error=str(exc))
        return ''

    if not result or not result[0]:
        return ''

    lines_data = result[0]

    items = []
    for line in lines_data:
        box, text_conf = line
        text = text_conf[0] if isinstance(text_conf, (list, tuple)) else str(text_conf)
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        y_center = sum(ys) / len(ys)
        x_center = sum(xs) / len(xs)
        items.append((y_center, x_center, text))

    if not items:
        return ''

    tolerance = 20
    items.sort(key=lambda t: t[0])
    buckets: List[List[tuple]] = []
    current_bucket: List[tuple] = []
    current_min_y = items[0][0]

    for item in items:
        if item[0] - current_min_y <= tolerance:
            current_bucket.append(item)
        else:
            if current_bucket:
                buckets.append(current_bucket)
            current_bucket = [item]
            current_min_y = item[0]

    if current_bucket:
        buckets.append(current_bucket)

    row_texts = []
    for bucket in buckets:
        bucket.sort(key=lambda t: t[1])
        row_text = ' '.join(t[2] for t in bucket)
        row_texts.append(row_text)

    return '\n'.join(row_texts)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Path) -> set:
    """Return set of (doc_filename, page_index) already processed."""
    done: set = set()
    if not cache_path.exists():
        return done
    with cache_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                done.add((entry['doc'], entry['page']))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _append_cache(cache_path: Path, doc: str, page: int, source: str, text: str) -> None:
    """Append one entry to the JSONL cache and flush immediately."""
    entry = {'doc': doc, 'page': page, 'source': source, 'text': text}
    with cache_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------------------
# Per-page pre-filter
# ---------------------------------------------------------------------------

def _get_native_text(page) -> Optional[str]:
    """
    Return native text if it is high-quality and long enough, else None.
    Uses is_low_quality_chunk from core.indexer.
    """
    text = page.get_text('text')
    if len(text) >= 500 and not is_low_quality_chunk(text):
        return text
    return None


# ---------------------------------------------------------------------------
# Main per-document processing function
# ---------------------------------------------------------------------------

def run_ocr(
    doc_path: Path,
    cache_path: Path,
    max_pages: Optional[int] = None,
    pages: Optional[List[int]] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict:
    """
    OCR (or native-extract) every page of *doc_path* and write results to
    *cache_path* (JSONL, one entry per page, resumable).

    Args:
        doc_path:  Path to the scanned PDF.
        cache_path: Path to the JSONL cache file.
        max_pages: If set, stop after processing this many pages (for testing).
        pages:     If set, only process these specific 0-indexed page numbers.
        progress_callback: Optional callable receiving a progress dict per page.
        should_stop: Optional callable; if it returns True, processing stops early.

    Returns:
        Summary dict with keys: doc, total_pages, ocr_count, native_count,
        skipped_cached, elapsed_s, stopped.
    """
    import fitz  # PyMuPDF — imported here so fitz is optional at module level

    start_time = time.time()
    doc_name = doc_path.name
    log.info("ocr_start", doc=doc_name, cache=str(cache_path))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_cache(cache_path)

    pdf = fitz.open(str(doc_path))
    total_pages = len(pdf)

    # Determine which page indices to process
    if pages is not None:
        page_indices = [p for p in pages if 0 <= p < total_pages]
    else:
        page_indices = list(range(total_pages))

    if max_pages is not None:
        page_indices = page_indices[:max_pages]

    count_ocr = 0
    count_native = 0
    done_count = 0
    skipped_cached = 0
    stopped = False

    for page_idx in page_indices:
        # Check stop flag before each page
        if should_stop is not None and should_stop():
            stopped = True
            break

        if (doc_name, page_idx) in done:
            log.debug("cache_hit", doc=doc_name, page=page_idx)
            skipped_cached += 1
            done_count += 1
            # Still emit progress for cache hits so UI can track total progress
            if progress_callback is not None:
                elapsed_s = time.time() - start_time
                progress_callback({
                    'doc': doc_name,
                    'page_index': page_idx,
                    'total_pages': total_pages,
                    'source': 'cached',
                    'ocr_count': count_ocr,
                    'native_count': count_native,
                    'done_count': done_count,
                    'elapsed_s': elapsed_s,
                })
            continue

        t0 = time.time()
        page = pdf[page_idx]

        native = _get_native_text(page)
        if native is not None:
            source = 'native'
            text = native
            count_native += 1
        else:
            source = 'ocr'
            try:
                img_bgr = _render_page(page, dpi=300)
                img_enhanced = _clahe_enhance(img_bgr)
                text = _ocr_image(img_enhanced)
            except Exception as exc:
                log.warning("page_ocr_error", doc=doc_name, page=page_idx, error=str(exc))
                text = ''
            count_ocr += 1

        elapsed_page = time.time() - t0
        done_count += 1
        elapsed_s = time.time() - start_time

        if progress_callback is None:
            print(
                f"DOC [{page_idx + 1}/{total_pages}] source={source} elapsed={elapsed_page:.1f}s"
            )
        else:
            progress_callback({
                'doc': doc_name,
                'page_index': page_idx,
                'total_pages': total_pages,
                'source': source,
                'ocr_count': count_ocr,
                'native_count': count_native,
                'done_count': done_count,
                'elapsed_s': elapsed_s,
            })

        _append_cache(cache_path, doc_name, page_idx, source, text)
        done.add((doc_name, page_idx))

    pdf.close()

    elapsed_s = time.time() - start_time

    log.info(
        "ocr_complete",
        doc=doc_name,
        pages_processed=done_count,
        ocr=count_ocr,
        native=count_native,
        stopped=stopped,
    )

    if progress_callback is None:
        print(
            f"\nSummary for {doc_name}: "
            f"{done_count} pages processed — "
            f"{count_ocr} OCR'd, {count_native} native (skipped OCR)"
        )
        if stopped:
            print("(Остановлено пользователем)")

    return {
        'doc': doc_name,
        'total_pages': total_pages,
        'ocr_count': count_ocr,
        'native_count': count_native,
        'skipped_cached': skipped_cached,
        'elapsed_s': elapsed_s,
        'stopped': stopped,
    }


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def ocr_stats(docs_dir: Path, cache_path: Path) -> List[Dict]:
    """
    Return per-PDF statistics for all PDFs in docs_dir.

    Each dict has keys:
        filename, total_pages, cached_pages, native_pages, ocr_pages,
        percent_done, is_scan_heavy
    """
    # Load cache into a structure keyed by doc filename
    cache_by_doc: Dict[str, List[Dict]] = {}
    if cache_path.exists():
        try:
            with cache_path.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        doc = entry.get('doc', '')
                        if doc not in cache_by_doc:
                            cache_by_doc[doc] = []
                        cache_by_doc[doc].append(entry)
                    except (json.JSONDecodeError, KeyError):
                        pass
        except OSError:
            pass

    # Collect all PDF paths, sorted by filename
    pdf_paths = sorted(docs_dir.glob('*.pdf'), key=lambda p: p.name)

    results = []
    for pdf_path in pdf_paths:
        filename = pdf_path.name

        # Get total pages
        total_pages = 0
        if pdf_path.exists():
            try:
                import fitz
                doc = fitz.open(str(pdf_path))
                total_pages = len(doc)
                doc.close()
            except Exception:
                total_pages = 0

        entries = cache_by_doc.get(filename, [])
        cached_pages = len(entries)
        native_pages = sum(1 for e in entries if e.get('source') == 'native')
        ocr_pages = sum(1 for e in entries if e.get('source') == 'ocr')
        percent_done = (cached_pages / total_pages * 100) if total_pages > 0 else 0.0
        is_scan_heavy = 'Databook' in filename

        results.append({
            'filename': filename,
            'total_pages': total_pages,
            'cached_pages': cached_pages,
            'native_pages': native_pages,
            'ocr_pages': ocr_pages,
            'percent_done': percent_done,
            'is_scan_heavy': is_scan_heavy,
        })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='OCR TrueBeam Field Service Databooks and write to a JSONL cache.'
    )
    parser.add_argument(
        '--doc',
        choices=['vol1', 'vol2', 'both'],
        default='both',
        help='Which volume(s) to process (default: both)',
    )
    parser.add_argument(
        '--cache',
        type=Path,
        default=Path('scripts/ocr_cache.jsonl'),
        help='Path to the JSONL cache file (default: scripts/ocr_cache.jsonl)',
    )
    parser.add_argument(
        '--docs-dir',
        type=Path,
        default=Path('docs'),
        help='Directory containing the PDF files (default: docs)',
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help='Process only the first N pages per doc (for testing)',
    )
    parser.add_argument(
        '--pages',
        type=str,
        default=None,
        help='Comma-separated 0-indexed page numbers to process, e.g. "415,100,250"',
    )
    return parser


DOC_FILENAMES = {
    'vol1': 'TrueBeam 3.0 Volume 1 Field Service Databook.pdf',
    'vol2': 'TrueBeam 3.0 Volume 2 Field Service Databook.pdf',
}


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    pages: Optional[List[int]] = None
    if args.pages:
        pages = [int(p.strip()) for p in args.pages.split(',') if p.strip()]

    docs_to_process = ['vol1', 'vol2'] if args.doc == 'both' else [args.doc]

    for vol_key in docs_to_process:
        doc_path = args.docs_dir / DOC_FILENAMES[vol_key]
        if not doc_path.exists():
            log.error("doc_not_found", path=str(doc_path))
            print(f"ERROR: PDF not found: {doc_path}")
            continue

        run_ocr(
            doc_path=doc_path,
            cache_path=args.cache,
            max_pages=args.max_pages,
            pages=pages,
        )


if __name__ == '__main__':
    main()
