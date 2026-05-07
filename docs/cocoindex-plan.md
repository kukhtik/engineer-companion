# CocoIndex Integration Plan — engineer-companion

## Goal
Replace custom `core/indexer.py` with CocoIndex declarative flow after MVP stabilization.

## Why
- Current `indexer.py` (~8KB) is handwritten and likely does full reindex on every run.
- CocoIndex provides incremental indexing: only new/changed PDF/doc files are processed.
- Removes boilerplate: parse → chunk → embed → LanceDB becomes a 30-line flow declaration.

## Current State
- `core/indexer.py` — custom indexer (PDF parsing, chunking, sentence-transformers embed, LanceDB write).
- `core/query.py` — search over LanceDB.
- Offline only: GGUF LLM + LanceDB, no cloud.

## Phases

### Phase 0 — MVP Freeze (now)
- Keep current `indexer.py` working.
- Do not add CocoIndex as dependency until docs set stabilizes.

### Phase 1 — Migration (after MVP)
- Deprecate `core/indexer.py`, replace with `core/cocoindex_bridge.py`.
- Flow: `LocalFile("assets/docs/**/*.pdf")` → `PDFParser` → `Chunker` → `Embed(multilingual-e5-small)` → `LanceDB`.
- Must stay offline: embed model via `sentence-transformers`, LanceDB local file.

### Phase 2 — Live Index (post-MVP)
- Watchdog on `assets/docs/` — incremental update when field engineer adds new Varian manual PDF.
- Agent can query "What is the calibration procedure for TrueBeam?" with fresh index.

## Constraints
- No cloud dependencies (Qdrant/Pinecone disallowed).
- Keep GGUF inference path untouched.
- `multilingual-e5-small` embedding model stays as-is.

## Flow Declaration (Phase 1)
```python
@cocoindex.flow_def(name="varian_docs")
def docs_flow(flow_builder):
    src = flow_builder.add_source(cocoindex.sources.LocalFile(patterns=["assets/docs/**/*.pdf"]))
    parsed = flow_builder.add_transform(cocoindex.transforms.PDFParser, src)
    chunked = flow_builder.add_transform(cocoindex.transforms.Chunker, parsed)
    embedded = flow_builder.add_transform(
        cocoindex.transforms.Embed(
            model="intfloat/multilingual-e5-small",
            local=True,  # sentence-transformers
        ),
        chunked,
    )
    flow_builder.add_sink(cocoindex.sinks.LanceDB, embedded, uri="assets/db/")
```

## Rejected Alternatives
- Qdrant/Pinecone — cloud, violates offline constraint.
- Rewrite `indexer.py` now — MVP unstable, docs set not fixed.
