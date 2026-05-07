"""Chaquopy bridge: expose core retrieval to Kotlin/Java.

No LLM here — only vector search (embeddings + LanceDB).
Kotlin handles LLM via MediaPipe.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Globals hold singletons across multiple bridge calls — Chaquopy reuses one interpreter.
_retriever: Any = None
_model: Any = None
_db_path: Path | None = None


def _ensure_init(db_path_str: str, embedding_model: str = "intfloat/multilingual-e5-small") -> dict[str, Any]:
    """Lazy init of sentence-transformers + LanceDB. Called internally before search."""
    global _retriever, _model, _db_path

    db_path = Path(db_path_str)
    if _db_path == db_path and _retriever is not None:
        return {"ok": True, "cached": True}

    try:
        from sentence_transformers import SentenceTransformer
        import lancedb

        logger.info("bridge_init", db_path=str(db_path), model=embedding_model)
        _model = SentenceTransformer(embedding_model, trust_remote_code=True, device="cpu")
        db = lancedb.connect(str(db_path))
        table = db.open_table("chunks")

        class _BridgeRetriever:
            def __init__(self, model, table, top_k: int = 8):
                self.model = model
                self.table = table
                self.top_k = top_k

            def search(self, query: str) -> list[dict]:
                emb = self.model.encode(query, normalize_embeddings=True, device="cpu").tolist()
                results = self.table.search(emb).metric("cosine").limit(self.top_k).to_list()
                out = []
                for r in results:
                    out.append({
                        "text": r.get("text", ""),
                        "source": r.get("source", ""),
                        "page": r.get("page", 0),
                        "section": r.get("section", ""),
                        "score": r.get("_distance", 0.0),
                    })
                return out

        _retriever = _BridgeRetriever(_model, table)
        _db_path = db_path
        return {"ok": True, "cached": False}

    except Exception as exc:
        logger.error("bridge_init_failed", error=str(exc))
        return {"ok": False, "error": str(exc)}


def search(query: str, db_path: str, embedding_model: str = "intfloat/multilingual-e5-small") -> str:
    """Kotlin calls this. Returns JSON string with results or error.

    Args:
        query: user question
        db_path: absolute path to LanceDB directory (e.g. /data/data/pkg/files/db/engineer.db)
        embedding_model: huggingface model name or absolute path to local model dir

    Returns:
        JSON string: {"ok": true, "results": [...]} or {"ok": false, "error": "..."}
    """
    init_result = _ensure_init(db_path, embedding_model)
    if not init_result["ok"]:
        return json.dumps({"ok": False, "error": init_result.get("error", "init failed")}, ensure_ascii=False)

    try:
        results = _retriever.search(query)
        return json.dumps({"ok": True, "results": results}, ensure_ascii=False)
    except Exception as exc:
        logger.error("bridge_search_failed", error=str(exc))
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


def get_db_info(db_path: str) -> str:
    """Return table row count and schema info."""
    try:
        import lancedb
        db = lancedb.connect(db_path)
        table = db.open_table("chunks")
        count = table.count_rows()
        return json.dumps({"ok": True, "row_count": count}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


# Entrypoint for direct CLI testing inside Chaquopy interpreter.
if __name__ == "__main__":
    import sys
    structlog.configure(
        processors=[structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(20),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    if len(sys.argv) >= 3:
        db = sys.argv[1]
        q = sys.argv[2]
        print(search(q, db))
    else:
        print("Usage: python chaquopy_bridge.py <db_path> <query>")
