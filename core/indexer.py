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
                 embedding_model_name: str = "intfloat/multilingual-e5-small") -> None:
        self.extractor = extractor
        self.chunker = chunker
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self._model = None
        self._db = None
        self._table = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
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
        chunks: list[Chunk] = []
        for page, heading, para in self.extractor.extract_with_structure(pdf_path):
            for chunk in self.chunker.chunk_paragraph(para, pdf_path.name, page, heading):
                chunks.append(chunk)
        if not chunks:
            logger.warning("no_text_extracted", path=str(pdf_path))
            return

        model = self._get_model()
        texts = [c.text for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

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
    ap.add_argument("--docs-dir", default="/mnt/e/DOCS", type=Path)
    ap.add_argument("--db-path", default="/mnt/e/engineer-companion/assets/db/engineer.db", type=Path)
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
