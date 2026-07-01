"""RAG Query Pipeline: embed query -> vector search -> rerank -> LLM answer"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import structlog

if TYPE_CHECKING:
    from core.cloud import OllamaCloudClient
    from core.cloud_usage import CloudUsageTracker

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Source-tag stripping (shared by local + cloud paths)
# ---------------------------------------------------------------------------

_SOURCE_TAG_RE = re.compile(r"\[ИСТОЧНИК\s*\d+(?:,\s*ИСТОЧНИК\s*\d+)*\]", re.IGNORECASE)


def _strip_source_tags(text: str) -> str:
    """Remove inline [ИСТОЧНИК N] citation tags from answer text."""
    return _SOURCE_TAG_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Query-time chunk quality filter
# ---------------------------------------------------------------------------
# Defense-in-depth filter that complements the indexing-time is_low_quality_chunk.
# Catches two classes of garbage that survive indexing:
#
#  (A) Very short chunks (< _MIN_TOKENS=5 tokens) — they bypass all indexing
#      heuristics yet rank into top-K via spurious keyword overlap (e.g. "MLC").
#
#  (B) Short-ish chunks with OCR-scan-artifact content that scores < 2 heuristics
#      at indexing time (e.g. 5-10 tokens of mixed codes and tildes).
#
# Additionally filters on section labels that are clearly OCR noise.

# Section labels that are clearly OCR noise: very short (≤3 chars), entirely
# non-alpha, or low alpha density (< 40%).  Legitimate headings like "MLC",
# "Steps", "Context", "About Interlocks" pass without issue.
_GC_SECTION_RE = re.compile(
    r"^[_\-•·~\s]{1,}$"       # only dashes/underscores/bullets/spaces/tildes
    r"|^.{1,2}$"               # ≤2 chars (single char, e.g. "j", "e", "0")
)


def _is_retrieval_garbage(text: str, section: str) -> bool:
    """Return True if this chunk should be dropped from the RAG context.

    Applies a layered check:
    1. Obvious garbage section labels (noise OCR headings — ≤2 chars, or all-symbol).
    2. Low alpha density in section (e.g. "DIMvaAricinERS(mm)", "'o", ":El").
    3. OCR scan-artifact patterns in text (tildes, tilde-combos, heavy underscores).
    4. Low alpha ratio in text (binary-like content that passed the 5-token indexing gate).

    Design note: we do NOT filter on text length alone because short but legitimate
    chunks like "● In the Service screen, choose MLC > Communications ." carry useful
    information.  The section label is the most reliable garbage signal for the corpus
    (all confirmed garbage chunks have symbolic/very-short section labels).
    """
    # --- 1. Section label: very short or all symbols ---
    sec = (section or "").strip()
    if sec and _GC_SECTION_RE.match(sec):
        return True
    # --- 2. Low alpha ratio in section label (catches "DIMvaAricinERS(mm)" etc.) ---
    if sec and len(sec) >= 4:
        alpha_sec = sum(1 for c in sec if c.isalpha())
        if alpha_sec / len(sec) < 0.35:
            return True

    # --- 3. OCR artifact patterns in text ---
    # Tildes/tilde-combos, repeated underscores — hallmarks of scanned
    # engineering drawings / databook pages that slipped through indexing.
    # Also catches text that starts with a tilde (scan drawing artifact).
    if re.search(r"~~|~{2,}|_{4,}|\.\.:|\bSCA~|^~", text.strip()):
        return True

    # --- 4. Low alpha ratio in text (mainly catches binary/code debris) ---
    if text:
        alpha_text = sum(1 for c in text if c.isalpha())
        if alpha_text / len(text) < 0.30:
            return True

    return False


# ---------------------------------------------------------------------------
# Conversational retrieval-query resolution
# ---------------------------------------------------------------------------
# Vector search embeds ONLY the current query text — conversation history is
# otherwise injected solely into the LLM prompt, never into the search step.
# A short follow-up that leans on anaphora ("она", "это") or an instruction
# with no technical noun at all ("подумай где она расположена, смотри
# чертежи") therefore has nothing for the embedder to anchor on, and vector
# search drifts to semantically-similar-sounding but topically unrelated
# chunks. Folding prior user turns into the retrieval query (NOT into the
# prompt's literal "ВОПРОС:" — that stays the verbatim question) restores
# the missing noun phrase cheaply, with no extra LLM call.
#
# The whole chat is considered, not just the last turn or two — a topic set
# early can still be the referent many turns later. But the anchor is capped
# by a total character budget (most recent turns kept in full, older ones
# dropped first) so a long chat doesn't dilute the embedding with stale topics.
_MAX_HISTORY_ANCHOR_CHARS = 200  # per-turn cap
_MAX_TOTAL_ANCHOR_CHARS = 600  # whole-anchor budget across all carried-forward turns


def _resolve_retrieval_query(query: str, history: list[dict] | None) -> str:
    if not history:
        return query
    q = query.strip()
    user_turns = [
        (t.get("content") or "").strip()[:_MAX_HISTORY_ANCHOR_CHARS]
        for t in history
        if t.get("role") == "user" and (t.get("content") or "").strip() != q
    ]
    if not user_turns:
        return query
    anchor_turns: list[str] = []
    budget = _MAX_TOTAL_ANCHOR_CHARS
    for turn in reversed(user_turns):
        if budget <= 0:
            break
        anchor_turns.append(turn[:budget])
        budget -= len(turn) + 1
    anchor_turns.reverse()
    anchor = " ".join(anchor_turns)
    return f"{anchor} {query}"


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
            from android.assets_loader import resolve_bundled_model
            local = resolve_bundled_model("embedder")
            if local is not None:
                model_id = str(local)
                extra: dict = {"local_files_only": True}
                logger.info("loading_embedding_model_local", path=model_id)
            else:
                model_id = self.embedding_model_name
                extra = {}
                logger.info("loading_embedding_model", model=model_id)
            self._model = SentenceTransformer(model_id, trust_remote_code=True, **extra)
        return self._model

    def _get_reranker(self):
        if not self.rerank_model_name:
            return None
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            from android.assets_loader import resolve_bundled_model
            local = resolve_bundled_model("reranker")
            if local is not None:
                model_id = str(local)
                logger.info("loading_reranker_local", path=model_id)
                self._reranker = CrossEncoder(model_id, local_files_only=True)
            else:
                logger.info("loading_reranker", model=self.rerank_model_name)
                self._reranker = CrossEncoder(self.rerank_model_name)
        return self._reranker

    def _get_table(self):
        if self._table is None:
            import lancedb
            db = lancedb.connect(str(self.db_path))
            self._table = db.open_table("chunks")
        return self._table

    def search(self, query: str, history: list[dict] | None = None) -> list[SearchResult]:
        model = self._get_model()
        retrieval_query = _resolve_retrieval_query(query, history)
        emb = model.encode(retrieval_query, normalize_embeddings=True, device="cpu").tolist()

        table = self._get_table()
        # Retrieve a larger pool so that after dropping garbage chunks there
        # are still enough clean candidates to rerank down to rerank_top_k.
        fetch_k = max(self.top_k, self.rerank_top_k * 6)
        results = table.search(emb).metric("cosine").limit(fetch_k).to_list()
        all_hits = [
            SearchResult(
                text=r["text"],
                source=r["source"],
                page=r["page"],
                section=r["section"],
                score=r.get("_distance", 0.0),
            )
            for r in results
        ]

        # Query-time quality filter: drop garbage before reranking/prompting.
        hits = [h for h in all_hits if not _is_retrieval_garbage(h.text, h.section)]
        n_dropped = len(all_hits) - len(hits)
        if n_dropped:
            logger.info(
                "retrieval_quality_filter",
                dropped=n_dropped,
                kept=len(hits),
                query=query[:120],
                retrieval_query=retrieval_query[:160] if retrieval_query != query else None,
            )
        # Keep at most top_k clean candidates for the reranker.
        hits = hits[:self.top_k]

        # Rerank if configured
        reranker = self._get_reranker()
        if reranker and hits:
            pairs = [(retrieval_query, h.text) for h in hits]
            scores = reranker.predict(pairs)
            ranked = [(s, h) for s, h in zip(scores, hits)]
            ranked.sort(key=lambda x: x[0], reverse=True)
            hits = [h for _, h in ranked[:self.rerank_top_k]]

        return hits

    def search_streaming(
        self, query: str, on_event: Callable[[dict], None], history: list[dict] | None = None
    ) -> list[SearchResult]:
        """Like search(), but fires progress events via on_event at key steps."""
        on_event({"stage": "embed"})
        model = self._get_model()
        retrieval_query = _resolve_retrieval_query(query, history)
        emb = model.encode(retrieval_query, normalize_embeddings=True, device="cpu").tolist()

        table = self._get_table()
        # Retrieve a larger pool so that after dropping garbage chunks there
        # are still enough clean candidates to rerank down to rerank_top_k.
        fetch_k = max(self.top_k, self.rerank_top_k * 6)
        results = table.search(emb).metric("cosine").limit(fetch_k).to_list()
        all_hits = [
            SearchResult(
                text=r["text"],
                source=r["source"],
                page=r["page"],
                section=r["section"],
                score=r.get("_distance", 0.0),
            )
            for r in results
        ]

        # Query-time quality filter: drop garbage before reranking/prompting.
        hits = [h for h in all_hits if not _is_retrieval_garbage(h.text, h.section)]
        n_dropped = len(all_hits) - len(hits)
        if n_dropped:
            logger.info(
                "retrieval_quality_filter",
                dropped=n_dropped,
                kept=len(hits),
                query=query[:120],
                retrieval_query=retrieval_query[:160] if retrieval_query != query else None,
            )
        # Keep at most top_k clean candidates for the reranker.
        hits = hits[:self.top_k]

        on_event({"stage": "search", "found": len(hits)})

        # Rerank if configured
        reranker = self._get_reranker()
        if reranker and hits:
            n_before = len(hits)
            pairs = [(retrieval_query, h.text) for h in hits]
            scores = reranker.predict(pairs)
            ranked = [(s, h) for s, h in zip(scores, hits)]
            ranked.sort(key=lambda x: x[0], reverse=True)
            hits = [h for _, h in ranked[:self.rerank_top_k]]
            on_event({"stage": "rerank", "from": n_before, "to": len(hits)})

        return hits


class PromptBuilder:
    """Builds prompt with system persona + retrieved context."""

    SYSTEM_PERSONA = (
        "Ты — эксперт-ассистент, отвечающий строго по предоставленной документации.\n"
        "ПРАВИЛА ОТВЕТА:\n"
        "1. Используй ТОЛЬКО те источники из КОНТЕКСТА, которые реально относятся к вопросу. "
        "Нерелевантные фрагменты — полностью игнорируй, не упоминай и не суммируй их.\n"
        "2. При цитировании указывай метку источника ТОЧНО так, как она написана в тексте контекста: "
        "например «[ИСТОЧНИК 2] Название документа стр.47». "
        "НЕ меняй название документа, не путай документы между собой, НЕ придумывай страницы.\n"
        "3. Если контекст релевантен вопросу, но не даёт точного дословного ответа — НЕ отказывайся сразу. "
        "Порассуждай на основе того, что есть: сопоставь фрагменты, сделай обоснованный вывод, предложи "
        "наиболее вероятный практический ответ. Чётко отделяй то, что взято дословно из источника "
        "(с точной ссылкой), от своего вывода/предположения — для вывода используй пометки вроде "
        "«судя по контексту», «вероятно», «прямо не указано, но исходя из...». "
        "Скажи «не найдено в документации» ТОЛЬКО если предоставленные фрагменты вообще не относятся "
        "к вопросу и не дают НИКАКОЙ зацепки для рассуждения.\n"
        "4. Отвечай на русском языке.\n"
        "5. НЕ выдумывай расшифровку аббревиатур и фактов, которых нет в контексте. "
        "Если точного определения аббревиатуры или термина нет в предоставленных фрагментах — "
        "скажи явно: «В документации найдено только косвенное упоминание» и приведи то, что есть. "
        "Никогда не изобретай расшифровку на основе догадок.\n"
        "КРИТИЧЕСКИ ВАЖНО: отвечай ТОЛЬКО на русском языке, даже если документация на английском. "
        "Весь текст ответа обязательно на русском; технические термины и названия моделей можно оставлять как есть."
    )

    def build(self, query: str, hits: list[SearchResult], history: list[dict] | None = None) -> str:
        context_parts: list[str] = []
        for i, h in enumerate(hits, 1):
            context_parts.append(
                f"[ИСТОЧНИК {i}] Документ: «{h.source}» стр.{h.page} раздел: {h.section}\n"
                f"{h.text}"
            )
        context = "\n\n".join(context_parts)

        history_block = ""
        if history:
            turns = history[-3:]
            lines = []
            for turn in turns:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user":
                    lines.append(f"Пользователь: {content[:300]}")
                elif role == "assistant":
                    lines.append(f"Ассистент: {content[:300]}")
            history_text = "\n".join(lines)
            if len(history_text) > 1000:
                history_text = history_text[-1000:]
            if history_text:
                history_block = f"\nПредыдущий диалог:\n{history_text}\n"

        prompt = (
            f"{self.SYSTEM_PERSONA}\n\n"
            f"КОНТЕКСТ:\n{context}\n"
            f"{history_block}\n"
            f"ВОПРОС: {query}\n\nОТВЕТ НА РУССКОМ ЯЗЫКЕ:"
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

    def ask(self, query: str, history: list[dict] | None = None) -> dict[str, Any]:
        hits = self.retriever.search(query, history=history)
        prompt = self.builder.build(query, hits, history=history)
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

    # ------------------------------------------------------------------
    # Cloud generation helpers
    # ------------------------------------------------------------------

    def _build_cloud_messages(
        self, query: str, hits: list[SearchResult], history: list[dict] | None
    ) -> list[dict[str, str]]:
        """Build OpenAI-style messages list for cloud chat API.

        Constructs:
          [system: SYSTEM_PERSONA]
          + optional history turns (alternating user/assistant, last 3 pairs)
          + [user: context block + question + RU steer]

        This intentionally mirrors PromptBuilder.build() so persona/context/steer
        are identical between local and cloud paths.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": PromptBuilder.SYSTEM_PERSONA}
        ]

        # History turns (mirror PromptBuilder.build logic: last 3 turns, ≤300 chars each)
        if history:
            for turn in history[-3:]:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content[:300]})

        # User message: context block + question + RU steer (matches PromptBuilder)
        context_parts: list[str] = []
        for i, h in enumerate(hits, 1):
            context_parts.append(
                f"[ИСТОЧНИК {i}] Документ: «{h.source}» стр.{h.page} раздел: {h.section}\n"
                f"{h.text}"
            )
        context = "\n\n".join(context_parts)

        user_content = (
            f"КОНТЕКСТ:\n{context}\n\n"
            f"ВОПРОС: {query}\n\nОТВЕТ НА РУССКОМ ЯЗЫКЕ:"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    def ask_streaming(
        self,
        query: str,
        on_event: Callable[[dict], None],
        history: list[dict] | None = None,
        # --- Cloud backend params (all optional; existing callers unaffected) ---
        backend: str = "local",
        cloud_model: str | None = None,
        cloud_client: "OllamaCloudClient | None" = None,
        usage_tracker: "CloudUsageTracker | None" = None,
    ) -> dict[str, Any]:
        """Stream a RAG answer, emitting pipeline events via on_event.

        Event shapes:
          {"stage": "embed"}
          {"stage": "search", "found": N}
          {"stage": "rerank", "from": N, "to": M}   # only if reranker is loaded
          {"stage": "prompt", "sources": N}
          {"stage": "generate_start"}
          {"stage": "token", "text": "<piece>"}
          {"stage": "done", "answer": "...", "sources": [...], ...}
          {"stage": "error", "message": "..."}        # on cloud errors

        Args:
            backend:       "local" (default) or "cloud".
            cloud_model:   Model name for cloud backend (required when backend="cloud").
            cloud_client:  OllamaCloudClient instance (required when backend="cloud").
            usage_tracker: CloudUsageTracker instance for recording cloud usage.
        """
        # --- Retrieval (identical for both backends) ---
        hits = self.retriever.search_streaming(query, on_event, history=history)
        context_meta = [
            {"source": h.source, "page": h.page, "section": h.section}
            for h in hits
        ]
        on_event({"stage": "prompt", "sources": len(hits)})

        # ================================================================
        # LOCAL backend (unchanged)
        # ================================================================
        if backend != "cloud":
            prompt = self.builder.build(query, hits, history=history)
            llm = self._get_llm()
            if llm is None:
                result: dict[str, Any] = {
                    "answer": "[LLM не загружена. Проверьте путь к модели.]",
                    "sources": context_meta,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
                on_event({"stage": "done", **result})
                return result

            on_event({"stage": "generate_start"})

            tokens: list[str] = []
            stream = llm.create_completion(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["</s>", "USER:", "ВОПРОС:"],
                stream=True,
            )
            for chunk in stream:
                chunk_text = chunk["choices"][0]["text"]
                tokens.append(chunk_text)
                on_event({"stage": "token", "text": chunk_text})

            full_answer = "".join(tokens).strip()
            result = {
                "answer": full_answer,
                "sources": context_meta,
                "prompt_tokens": 0,
                "completion_tokens": len(tokens),
            }
            on_event({"stage": "done", **result})
            return result

        # ================================================================
        # CLOUD backend
        # ================================================================
        from core.cloud import CloudError, CloudRateLimitError

        if cloud_client is None or cloud_model is None:
            err_result: dict[str, Any] = {
                "answer": "[Облачный клиент не настроен.]",
                "sources": context_meta,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
            on_event({"stage": "error", "message": "Облачный клиент не настроен."})
            on_event({"stage": "done", **err_result})
            return err_result

        messages = self._build_cloud_messages(query, hits, history)
        on_event({"stage": "generate_start"})

        tokens_cloud: list[str] = []
        usage_data: dict = {"prompt_tokens": 0, "completion_tokens": 0}

        def _on_done(stats: dict) -> None:
            usage_data["prompt_tokens"] = stats.get("prompt_tokens", 0)
            usage_data["completion_tokens"] = stats.get("completion_tokens", 0)

        try:
            for piece in cloud_client.chat_stream(
                model=cloud_model,
                messages=messages,
                on_done=_on_done,
            ):
                tokens_cloud.append(piece)
                on_event({"stage": "token", "text": piece})

        except CloudRateLimitError as exc:
            if usage_tracker is not None:
                usage_tracker.mark_rate_limited(retry_after_seconds=exc.retry_after)
            on_event({
                "stage": "error",
                "message": (
                    "Лимит Ollama Cloud исчерпан. Сброс позже; переключитесь на локальную модель."
                ),
            })
            rate_result: dict[str, Any] = {
                "answer": "",
                "sources": context_meta,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
            on_event({"stage": "done", **rate_result})
            return rate_result

        except CloudError as exc:
            on_event({"stage": "error", "message": f"Ошибка Ollama Cloud: {exc}"})
            cloud_err_result: dict[str, Any] = {
                "answer": "",
                "sources": context_meta,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
            on_event({"stage": "done", **cloud_err_result})
            return cloud_err_result

        # Success path
        if usage_tracker is not None:
            usage_tracker.record_request(
                prompt_tokens=usage_data["prompt_tokens"],
                completion_tokens=usage_data["completion_tokens"],
            )

        full_answer_cloud = _strip_source_tags("".join(tokens_cloud))
        cloud_result: dict[str, Any] = {
            "answer": full_answer_cloud,
            "sources": context_meta,
            "prompt_tokens": usage_data["prompt_tokens"],
            "completion_tokens": usage_data["completion_tokens"],
        }
        on_event({"stage": "done", **cloud_result})
        return cloud_result


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
