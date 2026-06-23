"""Indexer: Extract text from technical PDFs preserving heading hierarchy, chunk
with overlap, embed with sentence-transformers, and store in LanceDB."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

import fitz  # pymupdf
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Chunk quality filter (stdlib re only — no extra deps)
# Validated heuristic: score>=2 across 6 criteria flags garbage from scanned
# pages (binary debris, dot-trains, OCR gibberish, single-token fragments).
# False-positive rate on clean Russian/English technical text: ~0%.
# ---------------------------------------------------------------------------

_SCAN_ARTIFACT_RE = re.compile(
    r'[~]{3,}'
    r'|t-=-'
    r'|\|{2,}'
    r'|[=\-]{5,}'
    r'|(?<!\w)[^\w\s]{4,}(?!\w)'
    r'|[\x00-\x08\x0b\x0c\x0e-\x1f]'
)

_MIN_TOKENS = 5  # chunks with fewer whitespace-separated tokens are always dropped


def _qf_alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isalpha()) / len(text)


def _qf_nonascii_noncy_ratio(text: str) -> float:
    """Ratio of chars that are neither ASCII printable nor Cyrillic."""
    if not text:
        return 0.0
    def _is_ok(c: str) -> bool:
        cp = ord(c)
        if 0x20 <= cp <= 0x7E:
            return True
        if 0x0400 <= cp <= 0x04FF:
            return True
        if c in ' \t\n\r':
            return True
        return False
    return sum(1 for c in text if not _is_ok(c)) / len(text)


def _qf_avg_token_len(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(len(t) for t in tokens) / len(tokens)


def _qf_word_alpha_ratio(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    good = sum(
        1 for t in tokens
        if re.search(r'[a-zA-Za-zA-а-яА-ЯёЁ]{2,}', t)
    )
    return good / len(tokens)


def is_low_quality_chunk(text: str) -> bool:
    """Return True if *text* is garbage and should NOT be indexed.

    A chunk is garbage when it scores >=2 across 6 independent criteria
    derived from OCR/scan-artifact analysis of the corpus.  Clean Russian
    and English technical text cannot reach score>=2 (alpha_ratio ~0.9).

    Also returns True for chunks shorter than _MIN_TOKENS words, which are
    single-token debris too short to be useful for retrieval.
    """
    tokens = text.split()
    if len(tokens) < _MIN_TOKENS:
        return True

    score = 0

    # H1: very low alphabetic character density
    if _qf_alpha_ratio(text) < 0.40:
        score += 1

    # H2: high density of chars that aren't ASCII or Cyrillic
    if _qf_nonascii_noncy_ratio(text) > 0.15:
        score += 1

    # H3: average token length is absurdly long or suspiciously short
    atl = _qf_avg_token_len(text)
    if atl > 18 or (0 < atl < 1.5):
        score += 1

    # H4: known OCR/scan artifact patterns (tildes, control chars, etc.)
    if _SCAN_ARTIFACT_RE.search(text):
        score += 1

    # H5: almost no spaces (dense binary-like block)
    if text.count(' ') / len(text) < 0.03:
        score += 1

    # H6: most tokens contain no real alphabetic word
    if _qf_word_alpha_ratio(text) < 0.50:
        score += 1

    return score >= 2

@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    source: str  # "filename.pdf"
    page: int
    section: str  # heading breadcrumb, e.g. "3.2 | Interlocks"
    chunk_index: int
    tokens_estimate: int


def _estimate_tokens(text: str) -> int:
    # ~0.75 tokens per word for english technical prose; quick estimate
    return int(len(text.split()) * 0.75)


class PdfTextExtractor:
    def __init__(self, min_heading_size: float = 12.0) -> None:
        self.min_heading_size = min_heading_size

    def extract_with_structure(self, pdf_path: Path) -> Iterator[tuple[int, str, str]]:
        """Yield (page_number, heading_breadcrumb, paragraph_text)."""
        doc = fitz.open(str(pdf_path))
        current_heading = ""
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                paragraph_parts: list[str] = []
                for block in blocks:
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        for span in spans:
                            font_size = span.get("size", 10)
                            text = span.get("text", "").strip()
                            if not text:
                                continue
                            if font_size >= self.min_heading_size:
                                # Heading detected
                                if paragraph_parts:
                                    yield (page_num + 1, current_heading, " ".join(paragraph_parts))
                                    paragraph_parts.clear()
                                current_heading = text
                                continue
                            paragraph_parts.append(text)
                if paragraph_parts:
                    yield (page_num + 1, current_heading, " ".join(paragraph_parts))
        finally:
            doc.close()


class Chunker:
    def __init__(self,
                 chunk_size: int = 512,
                 overlap: int = 64,
                 max_heading_depth: int = 3) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_heading_depth = max_heading_depth

    def clean_heading(self, heading: str) -> str:
        # Normalize: collapse multiple spaces, strip page numbers that got mixed in
        heading = re.sub(r"\s+", " ", heading).strip()
        # Remove leading digits like "3.2 Title" stays, but standalone numbers go
        return heading

    def chunk_paragraph(self, text: str, source: str, page: int, heading: str) -> Iterator[Chunk]:
        words = text.split()
        stride = self.chunk_size - self.overlap
        chunk_index = 0
        for i in range(0, len(words), stride):
            slice_ = words[i:i + self.chunk_size]
            chunk_text = " ".join(slice_)
            yield Chunk(
                text=chunk_text,
                source=source,
                page=page,
                section=self.clean_heading(heading),
                chunk_index=chunk_index,
                tokens_estimate=_estimate_tokens(chunk_text),
            )
            chunk_index += 1
            if i + self.chunk_size >= len(words):
                break


class DocumentIndexPipeline:
    def __init__(self,
                 extractor: PdfTextExtractor,
                 chunker: Chunker,
                 db_path: Path,
                 embedding_model_name: str = "intfloat/multilingual-e5-small",
                 drop_low_quality: bool = True) -> None:
        self.extractor = extractor
        self.chunker = chunker
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.drop_low_quality = drop_low_quality
        self._model = None
        self._db = None
        self._table = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from android.assets_loader import resolve_bundled_model
            local = resolve_bundled_model("embedder")
            if local is not None:
                model_id = str(local)
                logger.info("loading_embedding_model_local", path=model_id)
                self._model = SentenceTransformer(model_id, trust_remote_code=True, local_files_only=True)
            else:
                logger.info("loading_embedding_model", model=self.embedding_model_name)
                self._model = SentenceTransformer(self.embedding_model_name, trust_remote_code=True)
        return self._model

    def _get_table(self):
        if self._table is None:
            import lancedb
            import pyarrow as pa
            self._db = lancedb.connect(str(self.db_path))
            if "chunks" not in self._db.table_names():
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("page", pa.int32()),
                    pa.field("section", pa.string()),
                    pa.field("chunk_index", pa.int32()),
                    pa.field("tokens", pa.int32()),
                    pa.field("vector", pa.list_(pa.float32(), 384)),
                ])
                self._table = self._db.create_table("chunks", schema=schema)
            else:
                self._table = self._db.open_table("chunks")
        return self._table

    def _compute_hash(self, chunk: Chunk) -> str:
        # Simple content fingerprint to avoid duplicates on re-runs
        payload = f"{chunk.source}:{chunk.page}:{chunk.chunk_index}:{chunk.text[:200]}"
        return hashlib.md5(payload.encode()).hexdigest()[:16]

    def index_pdf(self, pdf_path: Path) -> None:
        logger.info("indexing_pdf", path=str(pdf_path))
        all_chunks: list[Chunk] = []
        for page, heading, para in self.extractor.extract_with_structure(pdf_path):
            for chunk in self.chunker.chunk_paragraph(para, pdf_path.name, page, heading):
                all_chunks.append(chunk)
        if not all_chunks:
            logger.warning("no_text_extracted", path=str(pdf_path))
            return

        if self.drop_low_quality:
            kept: list[Chunk] = []
            dropped_count = 0
            for chunk in all_chunks:
                if is_low_quality_chunk(chunk.text):
                    dropped_count += 1
                else:
                    kept.append(chunk)
            logger.info(
                "quality_filter",
                path=str(pdf_path),
                total=len(all_chunks),
                kept=len(kept),
                dropped=dropped_count,
            )
            chunks = kept
        else:
            chunks = all_chunks

        if not chunks:
            logger.warning("all_chunks_filtered", path=str(pdf_path))
            return

        model = self._get_model()
        texts = [c.text for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True, device="cpu")

        table = self._get_table()
        records = []
        for chunk, emb in zip(chunks, embeddings):
            rec = {
                "id": self._compute_hash(chunk),
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "tokens": chunk.tokens_estimate,
                "vector": emb.tolist(),
            }
            records.append(rec)
        table.add(records)
        logger.info("pdf_indexed", path=str(pdf_path), chunks=len(records))

    def build_index(self, docs_dir: Path) -> None:
        pdfs = sorted(docs_dir.glob("*.pdf"))
        logger.info("found_pdfs", count=len(pdfs), docs_dir=str(docs_dir))
        for pdf in pdfs:
            self.index_pdf(pdf)
        logger.info("index_build_complete")

    def index_pdf_with_ocr(self, pdf_path: Path, ocr_cache: Path) -> None:
        """Index a PDF using pre-built OCR cache (from scripts/ocr_databooks.py).

        For each page in the OCR cache:
        - source=="native": use the native PyMuPDF text as-is
        - source=="ocr":    use the OCR-recovered text

        Chunks are built with the same Chunker and is_low_quality_chunk filter.
        OCR-recovered pages are tagged with section prefix "[OCR] " so they are
        distinguishable in retrieval results.

        Args:
            pdf_path:  Path to the scanned PDF (used only for its filename as source).
            ocr_cache: Path to JSONL cache produced by scripts/ocr_databooks.py.
        """
        import json as _json

        doc_name = pdf_path.name
        logger.info("indexing_pdf_with_ocr", path=str(pdf_path), cache=str(ocr_cache))

        if not ocr_cache.exists():
            logger.error("ocr_cache_not_found", cache=str(ocr_cache))
            return

        # Load all cache entries for this document, keyed by 0-indexed page number
        page_entries: dict[int, dict] = {}
        with ocr_cache.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                    if entry.get("doc") == doc_name:
                        page_entries[int(entry["page"])] = entry
                except (ValueError, KeyError):
                    pass

        if not page_entries:
            logger.warning("no_cache_entries_for_doc", doc=doc_name)
            return

        logger.info("loaded_ocr_cache", doc=doc_name, pages=len(page_entries))

        all_chunks: list[Chunk] = []
        for page_idx in sorted(page_entries):
            entry = page_entries[page_idx]
            text = entry.get("text", "").strip()
            if not text:
                continue
            source_tag = entry.get("source", "ocr")
            # 1-indexed page number for consistency with the rest of the pipeline
            page_num = page_idx + 1
            # Tag the section so OCR-recovered content is identifiable
            heading_prefix = "[OCR] " if source_tag == "ocr" else ""
            for chunk in self.chunker.chunk_paragraph(
                text, doc_name, page_num, heading_prefix
            ):
                all_chunks.append(chunk)

        if not all_chunks:
            logger.warning("no_chunks_from_ocr_cache", doc=doc_name)
            return

        if self.drop_low_quality:
            kept: list[Chunk] = []
            dropped_count = 0
            for chunk in all_chunks:
                if is_low_quality_chunk(chunk.text):
                    dropped_count += 1
                else:
                    kept.append(chunk)
            logger.info(
                "quality_filter",
                path=str(pdf_path),
                total=len(all_chunks),
                kept=len(kept),
                dropped=dropped_count,
            )
            chunks = kept
        else:
            chunks = all_chunks

        if not chunks:
            logger.warning("all_chunks_filtered_ocr", path=str(pdf_path))
            return

        model = self._get_model()
        texts = [c.text for c in chunks]
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cpu",
        )

        table = self._get_table()
        records = []
        for chunk, emb in zip(chunks, embeddings):
            rec = {
                "id": self._compute_hash(chunk),
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "tokens": chunk.tokens_estimate,
                "vector": emb.tolist(),
            }
            records.append(rec)
        table.add(records)
        logger.info("pdf_indexed_with_ocr", path=str(pdf_path), chunks=len(records))


if __name__ == "__main__":
    import argparse
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", default="docs", type=Path)
    ap.add_argument("--db-path", default="assets/db/engineer.db", type=Path)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    args = ap.parse_args()

    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = DocumentIndexPipeline(
        extractor=PdfTextExtractor(min_heading_size=11.0),
        chunker=Chunker(chunk_size=args.chunk_size, overlap=args.overlap),
        db_path=args.db_path,
    )
    pipeline.build_index(args.docs_dir)
