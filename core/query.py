"""RAG Query Pipeline: embed query -> vector search -> rerank -> LLM answer"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SearchResult:
    text: str
    source: str
    page: int
    section: str
    score: float


class Retriever:
    """Vector search + optional reranking via cross-encoder."""

    def __init__(
        self,
        db_path: Path,
        embedding_model_name: str = "intfloat/multilingual-e5-small",
        top_k: int = 8,
        rerank_top_k: int = 5,
        rerank_model: str = "",
    ) -> None:
        self.db_path = db_path
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.embedding_model_name = embedding_model_name
        self.rerank_model_name = rerank_model
        self._model = None
        self._reranker = None
        self._table = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("loading_embedding_model", model=self.embedding_model_name)
            self._model = SentenceTransformer(self.embedding_model_name, trust_remote_code=True)
        return self._model

    def _get_reranker(self):
        if not self.rerank_model_name:
            return None
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            logger.info("loading_reranker", model=self.rerank_model_name)
            self._reranker = CrossEncoder(self.rerank_model_name)
        return self._reranker

    def _get_table(self):
        if self._table is None:
            import lancedb
            db = lancedb.connect(str(self.db_path))
            self._table = db.open_table("chunks")
        return self._table

    def search(self, query: str) -> list[SearchResult]:
        model = self._get_model()
        emb = model.encode(query, normalize_embeddings=True, device="cpu").tolist()

        table = self._get_table()
        results = table.search(emb).metric("cosine").limit(self.top_k).to_list()
        hits = [
            SearchResult(
                text=r["text"],
                source=r["source"],
                page=r["page"],
                section=r["section"],
                score=r.get("_distance", 0.0),
            )
            for r in results
        ]

        # Rerank if configured
        reranker = self._get_reranker()
        if reranker and hits:
            pairs = [(query, h.text) for h in hits]
            scores = reranker.predict(pairs)
            ranked = [(s, h) for s, h in zip(scores, hits)]
            ranked.sort(key=lambda x: x[0], reverse=True)
            hits = [h for _, h in ranked[:self.rerank_top_k]]

        return hits


class PromptBuilder:
    """Builds prompt with system persona + retrieved context."""

    SYSTEM_PERSONA = (
        "Ты — сервисный инженер-эксперт по медицинским линейным ускорителям "
        "Varian TrueBeam и VitalBeam. Отвечай строго по предоставленной документации.\n"
        "ПРАВИЛА ОТВЕТА:\n"
        "1. Используй ТОЛЬКО те источники из КОНТЕКСТА, которые реально относятся к вопросу. "
        "Нерелевантные фрагменты (например, про прогрев рентгеновской трубки при вопросе об интерлоках) — "
        "полностью игнорируй, не упоминай и не суммируй их.\n"
        "2. При цитировании указывай метку источника ТОЧНО так, как она написана в тексте контекста: "
        "например «[ИСТОЧНИК 2] TrueBeam Instructions for Use стр.47». "
        "НЕ меняй название документа, НЕ путай TrueBeam с VitalBeam, НЕ придумывай страницы.\n"
        "3. Если ответ не найден ни в одном из релевантных фрагментов — честно скажи «не найдено в документации».\n"
        "4. Отвечай на русском языке."
    )

    def build(self, query: str, hits: list[SearchResult]) -> str:
        context_parts: list[str] = []
        for i, h in enumerate(hits, 1):
            # Include document name explicitly so model can copy it verbatim
            context_parts.append(
                f"[ИСТОЧНИК {i}] Документ: «{h.source}» стр.{h.page} раздел: {h.section}\n"
                f"{h.text}"
            )
        context = "\n\n".join(context_parts)
        prompt = (
            f"{self.SYSTEM_PERSONA}\n\n"
            f"КОНТЕКСТ:\n{context}\n\n"
            f"ВОПРОС: {query}\n\nОТВЕТ:"
        )
        return prompt


class RAGQueryPipeline:
    def __init__(
        self,
        db_path: Path,
        embedding_model_name: str = "intfloat/multilingual-e5-small",
        llm_model_path: Path | None = None,
        top_k: int = 20,
        rerank_top_k: int = 5,
        # Multilingual cross-encoder, ~120 MB, CPU-friendly (~1-2 s per 20 candidates).
        # Set rerank_model="" to disable reranking entirely.
        rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        max_tokens: int = 512,
        temperature: float = 0.3,
        llm_n_ctx: int = 4096,
        llm_n_threads: int | None = None,
    ) -> None:
        self.retriever = Retriever(
            db_path=db_path,
            embedding_model_name=embedding_model_name,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            rerank_model=rerank_model,
        )
        self.builder = PromptBuilder()
        self.llm_model_path = llm_model_path
        self.llm_n_ctx = llm_n_ctx
        self.llm_n_threads = llm_n_threads
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._llm = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if self.llm_model_path is None or not self.llm_model_path.exists():
            logger.error("llm_model_not_found", path=str(self.llm_model_path))
            return None
        from llama_cpp import Llama
        logger.info("loading_llm", model=str(self.llm_model_path))
        self._llm = Llama(
            model_path=str(self.llm_model_path),
            n_ctx=self.llm_n_ctx,
            n_threads=self.llm_n_threads,
            verbose=False,
        )
        return self._llm

    def ask(self, query: str) -> dict[str, Any]:
        hits = self.retriever.search(query)
        prompt = self.builder.build(query, hits)
        context_meta = [
            {"source": h.source, "page": h.page, "section": h.section}
            for h in hits
        ]

        llm = self._get_llm()
        if llm is None:
            return {
                "answer": "[LLM не загружена. Проверьте путь к модели.]",
                "sources": context_meta,
            }

        response = llm.create_completion(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["</s>", "USER:", "ВОПРОС:"],
        )
        answer = response["choices"][0]["text"].strip()
        return {
            "answer": answer,
            "sources": context_meta,
            "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
        }


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
    ap.add_argument("--db-path", default="/mnt/e/engineer-companion/assets/db/engineer.db", type=Path)
    ap.add_argument("--llm-path", default="/mnt/e/engineer-companion/assets/models/gemma-3-4b-it-Q4_K_M.gguf", type=Path)
    ap.add_argument("query", nargs="+")
    args = ap.parse_args()

    pipeline = RAGQueryPipeline(
        db_path=args.db_path,
        llm_model_path=args.llm_path if args.llm_path.exists() else None,
        llm_n_ctx=2048,
        llm_n_threads=2,
    )
    query = " ".join(args.query)
    result = pipeline.ask(query)
    print(json.dumps(result, ensure_ascii=False, indent=2))
