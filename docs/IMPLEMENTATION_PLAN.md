# IMPLEMENTATION PLAN — engineer-companion

**Agent:** deepseek-v4-flash
**Task:** Finish the project completely — Windows + Android clients, all tests passing, no loose ends.
**Model:** Gemma 4 4B IT (~2.3 GB, Q4_K_M GGUF available at `assets/models/`)
**Database:** 12 PDFs, 23757 chunks, 45 MB LanceDB (already built)
**Android:** Sideload APK (no Play Store), JNI + llama.cpp (no MediaPipe), Samsung S25 target (ARM64, 12 GB RAM)
**Git:** One commit per phase, mandatory before next phase.

---

## 0. State of the Art

| Component | Status | What's Missing |
|---|---|---|
| `core/indexer.py` | Done | No tests for `PdfTextExtractor`, `DocumentIndexPipeline` |
| `core/query.py` | Code ready | `llama-cpp-python` not installed → `ask()` never ran with real LLM |
| `windows/main_window.py` | UI done | Settings dialog, bookmarks filter, export chat, EXE build all missing |
| `tests/test_core.py` | 7 tests | No indexer/pipeline integration tests, no e2e LLM tests |
| `tests/test_behavioral_windows.py` | 12 tests | No settings/export/filter UI tests |
| `android/chaquopy/` | Skeleton | Chaquopy bridge exists but never tested on emulator or device |
| `android/main.py` | Kivy stub | Not part of final product — we use Jetpack Compose |
| `build_windows.py` / `.spec` | Scripts exist | Never executed, no EXE produced |
| `buildozer.spec` | Kivy config | Not used — replaced by Gradle + Chaquopy |

---

## 1. Phase — Dependency Setup & Environment Hardening

**Goal:** `.venv` with all working deps. `llama-cpp-python`, `torch` CPU, `sentence-transformers`, `lancedb`, `PySide6`.

### 1.1 Steps
1. `cd /mnt/e/engineer-companion`
2. Detect if `.venv` exists. If yes, backup: `mv .venv .venv.bak.$(date +%s)`.
3. Create `.venv` with Python 3.11:
   ```bash
   uv venv .venv --python 3.11
   source .venv/bin/activate
   ```
4. Install torch CPU ONLY (block CUDA):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
5. Install remaining deps:
   ```bash
   pip install sentence-transformers lancedb llama-cpp-python \
               pymupdf structlog numpy pyarrow pydantic \
               tiktoken PySide6 pytest pytest-qt pytest-asyncio \
               pyinstaller
   ```
6. Smoke-test imports:
   ```bash
   python -c "from llama_cpp import Llama; from sentence_transformers import SentenceTransformer; import lancedb; import PySide6; print('OK')"
   ```
7. **Verification checklist:**
   - [ ] `llama_cpp.Llama` loads the existing GGUF at `assets/models/` without crash
   - [ ] `SentenceTransformer('intfloat/multilingual-e5-small')` downloads and loads
   - [ ] `lancedb.connect(...).open_table('chunks')` succeeds on existing DB
   - [ ] `PySide6` import succeeds with offscreen platform

### 1.2 Tests to add
None (install phase). If smoke-test fails → fix before proceeding.

### 1.3 Expected commit message
```
fix(deps): install llama-cpp-python + torch CPU + all deps in .venv
```

---

## 2. Phase — Core Integration + LLM E2E Smoke

**Goal:** `RAGQueryPipeline.ask()` returns real answer from Gemma 4 4B IT. Benchmark CPU tok/s. Verify quality on medical question.

### 2.1 Steps
1. Run terminal benchmark:
   ```bash
   python core/query.py --query "What is a TrueBeam interlock error?"
   ```
2. Measure tokens/sec from `llama_cpp` verbose output.
3. If answer is garbage → inspect `PromptBuilder.SYSTEM_PERSONA` and chunk quality. Adjust `max_tokens`, `temperature` if needed.
4. Fix any runtime crash in `ask()`: GIL handling, `Llama` init args, stop tokens.

### 2.2 Tests to add (`tests/test_e2e.py`)
```python
def test_llm_smoke_loads_gguf():
    """Llama model loads from disk and answers minimally."""

def test_pipeline_ask_returns_answer():
    """RAGQueryPipeline.ask() returns non-empty answer + sources."""

def test_prompt_contains_citations():
    """Answer mentions source and page number."""

def test_indexer_pdf_produces_chunks():
    """PdfTextExtractor + Chunker produce >0 chunks for a real PDF."""

def test_pipeline_with_empty_query():
    """Graceful empty query handling."""
```

### 2.3 Verification checklist
- [ ] `test_e2e.py` all pass
- [ ] Answer quality acceptable on 3 technical questions
- [ ] tok/s benchmark logged in `docs/benchmark.md`

### 2.4 Expected commit message
```
feat(core): RAGQueryPipeline ask() e2e verified with Gemma 4 4B IT
```

---

## 3. Phase — Windows GUI Features

**Goal:** Settings dialog, bookmarks filter, export chat, search history filter.

### 3.1 Features detail

#### A. Settings Dialog (`windows/settings_dialog.py`)
- `QDialog` with:
  - Path picker for DB (`QLineEdit + QPushButton «Обзор»`)
  - Path picker for LLM model
  - `QSpinBox` for `temperature` (0.0–1.0, step 0.1)
  - `QSpinBox` for `max_tokens` (64–2048)
  - `QSpinBox` for `top_k` (1–20)
- Persist to `~/.engineer-companion/settings.json`.
- Load on app startup; apply to existing `RAGQueryPipeline` (recreate if paths change).

#### B. Bookmarks Filter + History Search
- Add `QCheckBox` «Только избранное» above chat log.
- When checked, `ChatHistory.format_html()` filters `bookmarked=True`.
- Add `QLineEdit` «Поиск по истории» — filters `entries` by substring match `query` or `answer`.
- History search is client-side only (no DB change).

#### C. Export Chat
- Add `QMenuBar` → «Файл → Экспорт в Markdown».
- Opens `QFileDialog` → writes `.md` with formatted Q/A/sources + `.json` backup.
- Export ALL history by default; if filter is active → export filtered subset.

#### D. UI Polish
- Add window icon (icon file at `assets/icon.ico` or `.png`).
- `QSplitter` sizes saved to settings and restored.

### 3.2 Tests to add (`tests/test_behavioral_windows.py`)
```python
class TestSettingsDialog:
    def test_settings_dialog_opens_and_saves()...
    def test_settings_persist_to_json()...
    def test_pipeline_recreated_on_model_path_change()...

class TestBookmarksFilter:
    def test_filter_shows_only_bookmarked()...
    def test_filter_off_shows_all()...

class TestHistorySearch:
    def test_search_by_query_substring()...
    def test_search_by_answer_substring()...
    def test_empty_search_shows_all()...

class TestExportChat:
    def test_export_creates_md_and_json()...
    def test_export_respects_current_filter()...
```

### 3.3 Verification checklist
- [ ] Settings dialog opens, values persist, restart restores
- [ ] Bookmarks filter toggles, star icons update
- [ ] History search finds by query and answer
- [ ] Export produces valid Markdown + JSON
- [ ] All new tests pass (target: 12 + 8 new = 20 behavioral tests)

### 3.4 Expected commit messages
```
feat(gui): Settings dialog with JSON persist
feat(gui): Bookmarks filter + history search UI
gui(export): File menu → Export chat to Markdown + JSON
```

---

## 4. Phase — Windows EXE Build

**Goal:** Working `dist/EngineerCompanion/` or single `.exe` runnable on Windows without Python.

### 4.1 Steps
1. Update `scripts/build_windows.py`:
   - Remove obsolete excludes (check torch version compatibility).
   - Ensure `assets/`, `core/`, `design/`, `windows/` bundled.
   - Add `--hidden-import` for any newly used packages.
2. Run `./run.sh build` or `python scripts/build_windows.py`.
3. Fix PyInstaller missing-module errors until build succeeds.
4. Test EXE on Windows host (double-click from `dist/`):
   - App opens
   - Search returns answer
   - Settings persist
   - No terminal window pops up (`--windowed`)
5. Update `.gitignore` for `build/` and `dist/`.

### 4.2 Tests
- No new pytest tests. Verification is manual on Windows host.
- Log success/failure in `docs/build_log.md`.

### 4.3 Expected commit message
```
feat(build): working Windows EXE via PyInstaller
```

---

## 5. Phase — Android Native Bridge (JNI + llama.cpp)

**Goal:** Android app uses Chaquopy for retrieval + JNI for LLM inference with Gemma 4 4B IT. Verified on emulator API 34 + Samsung S25.

### 5.1 Architecture Decision — JNI (not MediaPipe)
- We use llama.cpp compiled to `libllama.so` for ARM64.
- JNI wrapper `native-lib.cpp` exposes `loadModel(path)`, `generate(prompt, maxTokens, temperature)` to Kotlin.
- Kotlin (`MainActivity`) → `PythonAdapter.search()` → results → Kotlin builds prompt → JNI `generate()` → streaming to Compose.

### 5.2 File structure (new + modified)
```
android/app/src/main/
├── cpp/native-lib.cpp          # JNI bridge to llama.cpp
├── CMakeLists.txt              # Build libllama.so
├── java/com/varian/engcomp/
│   ├── MainActivity.kt         # Compose theme + navigation
│   ├── PythonAdapter.kt        # ← modify: drop MediaPipe refs
│   ├── LlamaEngine.kt          # JNI wrapper
│   ├── model/
│   │   ├── SearchResult.kt
│   │   └── ChatMessage.kt
│   ├── ui/
│   │   ├── SearchScreen.kt
│   │   ├── AnswerPanel.kt
│   │   ├── SourcesList.kt
│   │   └── theme/
│   │       └── DribbbleDark.kt
│   └── viewmodel/
│       └── SearchViewModel.kt
├── python/
│   └── android/
│       ├── __init__.py
│       ├── assets_loader.py
│       └── chaquopy/
│           └── chaquopy_bridge.py  # ← keep, verify on emulator
└── assets/
    ├── db/engineer.db/           # LanceDB
    └── models/
        └── gemma-3-4b-it-Q4_K_M.gguf  # ← replace with Gemma 4 4B
```

### 5.3 Gradle build config
- `build.gradle` must enable `externalNativeBuild` with CMake.
- Chaquopy pip block: `sentence-transformers`, `lancedb`, `numpy`, `pyarrow`, `structlog` (NO `llama-cpp-python` — it's replaced by JNI).
- `minSdk 28`, `targetSdk 34`, `abiFilters "arm64-v8a"` (S25 only).
- `sourceSets.main.jniLibs.srcDirs` must include `src/main/jniLibs/arm64-v8a/`.

### 5.4 Steps
1. Download/build llama.cpp for Android ARM64:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp
   cd /tmp/llama.cpp
   mkdir build-android && cd build-android
   cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DLLAMA_BUILD_EXAMPLES=OFF ..
   make -j4
   # Produces: libllama.so, libggml.so
   ```
2. Copy `.so` files to `android/app/src/main/jniLibs/arm64-v8a/`.
3. Write `native-lib.cpp` with minimal JNI:
   - `Java_com_varian_engcomp_LlamaEngine_loadModel(JNIEnv*, jobject, jstring path)` → `llama_load_model_from_file`
   - `Java_com_varian_engcomp_LlamaEngine_generate(JNIEnv*, jobject, jstring prompt, jint maxTokens, jfloat temp)` → token-by-token generation, return `jstring`
4. Write `LlamaEngine.kt` wrapping JNI calls. Mark `@Throws(IllegalStateException::class)`.
5. Write `SearchViewModel.kt`:
   - `search(query)` → `PythonAdapter.search()` → `SearchResult` list
   - `generateAnswer(prompt)` → `LlamaEngine.generate()` → `LiveData<String>` for streaming
6. Modify `PythonAdapter.kt`: remove MediaPipe imports, keep Chaquopy search only.
7. Write Jetpack Compose UI:
   - `SearchScreen` — input, button, lazy list of sources
   - `AnswerPanel` — streaming text display with scroll
   - `DribbbleDark.kt` — Material3 dark theme tokens mapping
8. Copy `assets/db/engineer.db/` into `src/main/assets/db/`.
9. Copy GGUF into `src/main/assets/models/`.
10. Build:
    ```bash
    ./gradlew assembleDebug
    ```
11. Install on emulator API 34:
    ```bash
    adb install app/build/outputs/apk/debug/app-debug.apk
    ```
12. Test cold start (copies assets, Python init, model load).
13. Test search + answer (no crash, text appears).
14. Move to S25, repeat.

### 5.5 Android tests to add
```
android/app/src/test/java/com/varian/engcomp/
├── PythonAdapterUnitTest.kt     # Mock chaquopy search
├── LlamaEngineUnitTest.kt       # JNI mock / load model test
└── PromptBuilderUnitTest.kt     # Verify prompt format
android/app/src/androidTest/java/com/varian/engcomp/
├── SearchFlowInstrumentedTest.kt  # Espresso/Compose test: type → search → answer visible
└── OfflineInstrumentedTest.kt     # Disable wifi → still works
```

### 5.6 Verification checklist
- [ ] APK > 100 MB (sideload OK, no Play Store)
- [ ] Emulator: search returns sources
- [ ] Emulator: LLM generates non-empty answer
- [ ] S25: same result, speed acceptable (>1 tok/s)
- [ ] APi > 34 instrumented tests pass
- [ ] App survives airplane mode

### 5.7 Expected commit messages
```
feat(android): JNI bridge libllama.so for Gemma 4 4B IT
feat(android): Jetpack Compose UI SearchScreen + AnswerPanel
feat(android): Gradle + Chaquopy + llama.cpp CMake build
test(android): unit + instrumented tests for search and LLM
```

---

## 6. Phase — Test Coverage Audit

**Goal:** Every module covered. Target: 30+ tests core, 20+ behavioral, 10+ Android.

### 6.1 Windows test matrix

| Module | Tests | File |
|---|---|---|
| `PdfTextExtractor` | `test_extracts_headings`, `test_empty_pdf` | `test_core.py` |
| `Chunker.chunk_paragraph` | `test_long_paragraph_splits`, `test_overlap` | `test_core.py` |
| `DocumentIndexPipeline` | `test_index_pdf_adds_to_table`, `test_build_index_counts` | `test_core.py` |
| `PromptBuilder` | `test_build_includes_query`, `test_empty_hits` | `test_core.py` |
| `Retriever.search` | `test_search_returns_results`, `test_rerank_filter` | `test_core.py` |
| `RAGQueryPipeline.ask` | `test_ask_with_mock_llm`, `test_ask_model_missing` | `test_e2e.py` |
| `LLM real smoke` | `test_llm_loads_gguf`, `test_answer_not_empty` | `test_e2e.py` |
| `ChatHistory` | `test_add`, `test_toggle`, `test_clear`, `test_prune`, `test_format_html` | `test_behavioral_windows.py` |
| `CompanionWindow search` | `test_user_search_happy_path`, `test_search_without_pipeline` | `test_behavioral_windows.py` |
| `Settings dialog` | `test_opens`, `test_saves`, `test_applied` | `test_behavioral_windows.py` |
| `Bookmarks filter` | `test_filter_on`, `test_filter_off` | `test_behavioral_windows.py` |
| `History search` | `test_search_query`, `test_search_answer`, `test_empty` | `test_behavioral_windows.py` |
| `Export chat` | `test_export_md_exists`, `test_export_json_exists` | `test_behavioral_windows.py` |
| `UI edge` | `test_concurrent_search_disabled`, `test_error_displayed` | `test_behavioral_windows.py` |

### 6.2 Android test matrix

| Module | Tests | File |
|---|---|---|
| `PythonAdapter.search` | `test_returns_results`, `test_error_empty` | `PythonAdapterUnitTest.kt` |
| `LlamaEngine` | `test_loadModel_success`, `test_generate_not_empty` | `LlamaEngineUnitTest.kt` |
| `PromptBuilder` | `test_format_includes_sources` | `PromptBuilderUnitTest.kt` |
| `SearchScreen` | `test_type_query`, `test_result_list_populates` | `SearchFlowInstrumentedTest.kt` |
| `AnswerPanel` | `test_answer_streams`, `test_markdown_rendered` | `SearchFlowInstrumentedTest.kt` |
| `Offline` | `test_airplane_mode_search` | `OfflineInstrumentedTest.kt` |

### 6.3 Verification checklist
- [ ] `pytest tests/` → all pass (target: 30+)
- [ ] `./gradlew test` → Android unit tests pass
- [ ] `./gradlew connectedCheck` → instrumented tests pass (emulator or S25)

### 6.4 Expected commit message
```
test: full coverage — core 30, behavioral 20, android 10+
```

---

## 7. Phase — Documentation + Git + Delivery

### 7.1 Update `README.md`
- Windows install: download EXE, run.
- Windows dev: clone, `./run.sh test`, `./run.sh gui`.
- Android install: sideload APK + copy assets, or single APK with bundled DB+GGUF.
- Android dev: Android Studio + NDK + CMake, `./gradlew assembleDebug`.

### 7.2 Delivery artifacts
```
dist/
├── EngineerCompanion-windows-x64.zip    # PyInstaller output
└── EngineerCompanion-android-v0.1.0.apk   # Gradle assembleDebug/Release
```

### 7.3 Git finalization
- Tag: `git tag -a v0.1.0 -m "Windows + Android complete"`
- Push tag.

### 7.4 Verification checklist
- [ ] README has install steps for both platforms
- [ ] Tag v0.1.0 pushed
- [ ] No uncommitted files (`git status --short` empty)

### 7.5 Expected commit message
```
docs: README + build artifacts + v0.1.0 tag
```

---

## 8. Summary — All Phases

| Phase | Work | Tests added | Git commit |
|---|---|---|---|
| 1 | Install deps | Smoke imports | `fix(deps): ...` |
| 2 | Core LLM e2e | `test_e2e.py` (4–6) | `feat(core): ...` |
| 3 | Windows GUI features | `test_behavioral_windows.py` (+8) | 3 commits |
| 4 | Windows EXE | Manual test | `feat(build): ...` |
| 5 | Android JNI + Compose | Android unit + instrumented (10+) | 4 commits |
| 6 | Coverage audit | Fill gaps | `test: full coverage` |
| 7 | Docs, tag, artifacts | — | `docs: README + v0.1.0` |

**Final target:**
- Windows: `EngineerCompanion.exe` or folder, all GUI features, tests green.
- Android: `app-debug.apk` or signed release, search + answer on S25, tests green.
- No missing tests. No uncommitted code.
