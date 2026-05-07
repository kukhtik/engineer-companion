# MVP Plan — engineer-companion

## Completed
- [x] `pyproject.toml` — зависимости declarative
- [x] `core/indexer.py` default `--docs-dir` → repo-relative `docs/`
- [x] `.gitignore` — `docs/`, `assets/db/`, `assets/models/`, `.venv/`
- [x] Phase A — Index build: 12 PDF, 23757 chunks, 45 MB LanceDB
- [x] Phase B — Query pipeline: `Retriever.search()` returns results with source/page/section
- [x] Phase C — UI tests: 9/9 passed (headless offscreen)
- [x] Phase D — GUI skeleton: search, results, chat panel

## In Progress / Blocked
- [ ] End-to-end with LLM (`llama-cpp-python` install blocked)
- [ ] Core unit tests (indexer + query, без UI)

## Todo
1. **Environment fix** — установить `llama-cpp-python`, `torch` CPU, `sentence-transformers` в `.venv`
2. **End-to-end test** — `RAGQueryPipeline.ask("TrueBeam interlock")` с Gemma 3 4B
3. **Core unit tests**
   - `test_indexer_extracts_text_from_pdf`
   - `test_indexer_chunks_have_overlap`
   - `test_indexer_heading_detection`
   - `test_retriever_search_returns_searchresult`
   - `test_retriever_cosine_metric_sorts_correctly`
   - `test_prompt_builder_includes_system_persona`
   - `test_prompt_builder_includes_source_citations`
   - `test_rag_pipeline_ask_returns_dict_with_answer_and_sources`
4. **GUI enhancements**
   - Rich text ответов (QTextBrowser с markdown-like styling)
   - История чата (список сессий, сохранение в SQLite)
   - Избранное/закладки (star результат, экспорт в JSON)
5. **Entry point**
   - `run.sh` / `run.bat` — активация `.venv` + `python windows/main_window.py`
   - `python -m engineer_companion` (пакетный entry point)
6. **Git**
   - Commit: deps, path fix, `.gitignore`, core, windows, tests
   - Tag: `v0.1-mvp`

## Constraints
- No cloud (Qdrant/Pinecone disallowed).
- GGUF inference — локально, через `llama-cpp-python`.
- `multilingual-e5-small` embedding model stays.
- CPU-only на данном хосте (GTX 1050 sm_61 несовместим с PyTorch CUDA 12.6).
