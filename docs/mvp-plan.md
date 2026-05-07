# MVP Plan — engineer-companion

## Completed
- [x] `pyproject.toml` — зависимости declarative + entry points (`engineer-companion`, `engineer-indexer`)
- [x] `core/indexer.py` default `--docs-dir` → repo-relative `docs/`
- [x] `.gitignore` — `docs/`, `assets/db/`, `assets/models/`, `.venv/`
- [x] Phase A — Index build: 12 PDF, 23757 chunks, 45 MB LanceDB
- [x] Phase B — Query pipeline: `Retriever.search()` returns results with source/page/section
- [x] Phase C — Core unit tests: 7/7 passed (`test_core.py`)
- [x] Phase D — UI tests: 12/12 passed (`test_behavioral_windows.py` headless offscreen)
- [x] Phase E — GUI: QTextBrowser + markdown-ish rendering, ChatHistory persistence (`~/.engineer-companion/history.json`)
- [x] Phase F — Bookmarks: ☆/★ toggle, anchor click in QTextBrowser, auto-prune >200
- [x] Phase G — Clear/Prune UI buttons: clear all, manual prune old non-bookmarked
- [x] `run.sh` launcher — `gui`, `test`, `index`, `run` modes
- [x] 19/19 tests green, CI-ready

## In Progress / Planned (Next 5 Steps)
1. **Smoke test LLM e2e** — `RAGQueryPipeline.ask("TrueBeam interlock")` с `gemma-3-4b-it-Q4_K_M.gguf`, benchmark CPU tok/s, verify answer quality.
2. **Bookmarks filter + search history** — UI checkbox «Только избранное», QLineEdit поиск по `query`/`answer` в `history.json`.
3. **Build skills** —
   - **Windows EXE** (`scripts/build_windows.py`): PyInstaller однофайл, `--windowed`, exclude torch (CPU-only), assets/data bundled, icon.
   - **Android APK** (`scripts/build_android.py`): Chaquopy bridge, buildozer spec, test on host Android Studio.
4. **Settings dialog** — `QDialog` для путей (db, model), `temperature`, `max_tokens`, `top_k` вместо CLI-аргументов; persist в `~/.engineer-companion/settings.json`.
5. **Export chat** — меню «Файл → Экспорт в Markdown» с вопросами, ответами, источниками; сохранить `.md` + `.json` backup.

## Constraints
- No cloud (Qdrant/Pinecone disallowed).
- GGUF inference — локально, через `llama-cpp-python`.
- `multilingual-e5-small` embedding model stays.
- CPU-only на данном хосте (GTX 1050 sm_61 несовместим с PyTorch CUDA 12.6).
