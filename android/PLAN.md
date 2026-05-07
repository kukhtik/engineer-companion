# Android MVP Plan

## Objective
Offline RAG companion on Samsung S25 (ARM64, 12GB RAM, NPU via MediaPipe).

## Architecture
Hybrid: **Jetpack Compose UI + Chaquopy Python bridge + MediaPipe LLM Inference API**.

- **Python core** (`core/`) — retrieval only: embeddings + LanceDB vector search. Runs inside Chaquopy.
- **MediaPipe** — LLM inference on NPU/GPU. Loads GGUF from internal storage.
- **Kotlin** — orchestrates: UI → Chaquopy retrieval → MediaPipe generation → UI update.

## Why not pure Python (Kivy/python-for-android)?
`llama-cpp-python` requires C++ compilation for ARM64 with GPU/NPU acceleration. python-for-android recipes are brittle for PyTorch + LanceDB + llama-cpp. Chaquopy + native MediaPipe is more maintainable.

## Phase 1: Core Retrieval Bridge (Chaquopy)

### 1.1 Asset Pipeline
- `assets/models/` → `src/main/assets/models/` (APK raw assets)
- `assets/db/` → `src/main/assets/db/`
- On first launch: copy from APK assets to app internal storage (`/data/data/<pkg>/files/`)
- Chaquopy Python sees paths via `context.getFilesDir()`

### 1.2 Python Bridge (`chaquopy_bridge.py`)
- Expose `search(query: str) -> List[Dict]` to Kotlin
- No LLM loading — retrieval only
- Lazy model loading: `SentenceTransformer` + `LanceDB` initialized on first call
- Thread-safe: GIL protects Python calls from Chaquopy

### 1.3 Kotlin Adapter (`PythonAdapter.kt`)
- `init(context)` — start Python, verify assets copied
- `search(query)` → `chaquopy_bridge.search(query)` → JSON string → Kotlin parse
- `close()` — cleanup

### 1.4 Android Module Dependencies
```
// Chaquopy
id 'com.chaquo.python' version '16.0.0'

// MediaPipe LLM Inference
implementation 'com.google.mediapipe:tasks-genai:0.10.14'

// Jetpack Compose
implementation platform('androidx.compose:compose-bom:2024.06.00')
```

## Phase 2: LLM Inference (MediaPipe)

### 2.1 Model
- Target: Gemma 3 4B IT (Q4_K_M, ~2.5GB) — fits Samsung S25 RAM budget
- Alternative: Qwen 2.5 7B Q4_K_M (~4.5GB) — heavier, fallback
- Convert GGUF → MediaPipe format if needed, or use `LlamaInference` task directly with GGUF

### 2.2 MediaPipe Integration
- `LlmInference.createFromFile(context, modelPath)`
- GPU delegate: `BaseOptions.builder().setDelegate(Delegate.GPU)`
- Samsung NPU may activate automatically via GPU delegate on Adreno

### 2.3 Prompt Pipeline
1. Kotlin gets query from user
2. Kotlin → `PythonAdapter.search(query)` → gets `List<SearchResult>`
3. Kotlin builds prompt (`PromptBuilder.SYSTEM_PERSONA + context + query`)
4. Kotlin → `LlmInference.generateResponse(prompt)` → streams tokens
5. Kotlin → Compose UI with streaming text

## Phase 3: Jetpack Compose UI

### 3.1 Screens
- **SearchScreen**: search bar, results list, answer panel
- **SourcesScreen**: tapped source detail (full text snippet)
- **SettingsScreen**: model info, storage usage, re-index trigger

### 3.2 Design Tokens
Apply `DribbbleDarkCSS` from `design/tokens.py` as Compose Material3 dark theme:
- `bg_base` → `Color(0xFF0A0A0A)`
- `accent_cyan` → `Color(0xFF3EB8B5)`
- `text_primary` → `Color(0xFFF4F3F4)`

### 3.3 Offline Indicator
Persistent banner: "Работает offline — модели и документы локальны"

## Phase 4: Testing

- **Unit**: `PythonAdapter` mock, `PromptBuilder` prompt format assertions
- **Integration**: Chaquopy + real LanceDB on emulator (API 34)
- **Device**: Samsung S25 — cold start, NPU utilization check via `adb shell dumpsys gpu`

## Constraints & Risks

| Risk | Mitigation |
|------|------------|
| MediaPipe doesn't support target GGUF | Fallback to CPU inference via `LlamaInference` with CPU delegate |
| Chaquopy + LanceDB binary wheels | Test `pip install lancedb` in Chaquopy; if fails, build from source with `chaquopy { pip { options "--no-binary" } }` |
| APK size > 100MB (Play Store limit) | Bundle: base app ~20MB, models via Play Feature Delivery or direct download on first launch |
| Samsung NPU not accessible | GPU delegate is sufficient; Adreno 750 handles 4B models at acceptable speed |

## Build Order

1. `chaquopy_bridge.py` + `PythonAdapter.kt` → test retrieval on emulator
2. `LlmInference` integration with small model (Gemma 2B test) → validate streaming
3. Compose UI shell → connect search → retrieval → generation flow
4. Full model + all docs → device test
5. APK signing + distribution (sideload or internal Play track)

## Fallback Plan

If Chaquopy fails with LanceDB or MediaPipe fails with GGUF:
- Switch to **Flutter + FFI** with Rust core (`candle` for LLM, `hnsw` for vectors)
- Or: Android WebView + WASM build of llama.cpp (slower, but no native compilation issues)

## Files Created
- `android/chaquopy/chaquopy_bridge.py`
- `android/chaquopy/PythonAdapter.kt`
- `android/chaquopy/README.md`
- `android/main.py` (Kivy fallback, asset-resolver ready)
- `android/assets_loader.py` (shared)
