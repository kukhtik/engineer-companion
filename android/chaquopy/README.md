# Chaquopy Build Instructions

## Prerequisites
- Android Studio Hedgehog+ (API 34 SDK)
- Python 3.11 (matching Chaquopy default ABI)
- `git-lfs` если модели в репозитории

## Step 1: Project Setup

### `build.gradle` (Project level)
```gradle
plugins {
    id 'com.android.application' version '8.5.0' apply false
    id 'com.chaquo.python' version '16.0.0' apply false
}
```

### `build.gradle` (Module: app)
```gradle
plugins {
    id 'com.android.application'
    id 'com.chaquo.python'
}

android {
    namespace 'com.varian.engineercompanion'
    compileSdk 34

    defaultConfig {
        applicationId "com.varian.engineercompanion"
        minSdk 28
        targetSdk 34
        versionCode 1
        versionName "0.1.0"

        ndk {
            abiFilters "arm64-v8a"  // Samsung S25 only
        }
    }

    // Chaquopy: configure Python dependencies
    chaquopy {
        defaultConfig {
            version "3.11"
            buildPython "/usr/bin/python3.11"

            pip {
                install "sentence-transformers==3.0.1"
                install "lancedb==0.12.0"
                install "numpy==1.26.4"
                install "pyarrow==15.0.2"
                install "structlog==24.4.0"
                // llama-cpp-python is intentionally excluded — LLM runs via MediaPipe native
            }
        }
    }

    buildTypes {
        release {
            minifyEnabled false
        }
    }
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_17
        targetCompatibility JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = '17'
    }
}

dependencies {
    implementation 'com.google.mediapipe:tasks-genai:0.10.14'
    implementation platform('androidx.compose:compose-bom:2024.06.00')
    implementation 'androidx.compose.ui:ui'
    implementation 'androidx.compose.material3:material3'
}
```

## Step 2: Assets

Copy from host repo to Android assets:
```bash
# From repo root on host
mkdir -p android/app/src/main/assets/db
mkdir -p android/app/src/main/assets/models

cp assets/db/engineer.db android/app/src/main/assets/db/
cp assets/models/* android/app/src/main/assets/models/   # GGUF, embeddings model dir
```

> **Note**: APK size limit is 100MB for Play Store. If models exceed this, use Play Feature Delivery or download on first launch.

## Step 3: Source Layout

```
app/src/main/
├── assets/
│   ├── db/
│   │   └── engineer.db
│   └── models/
│       └── gemma-3-4b-it-q4_k_m.gguf
├── java/com/varian/engineercompanion/
│   ├── MainActivity.kt
│   ├── PythonAdapter.kt          # ← copied from android/chaquopy/
│   ├── ui/
│   │   ├── SearchScreen.kt
│   │   ├── AnswerPanel.kt
│   │   └── theme/
│   │       └── DribbbleDark.kt   # Material3 theme from tokens
│   └── viewmodel/
│       └── SearchViewModel.kt
└── python/
    └── android/
        ├── __init__.py
        ├── assets_loader.py      # ← copied from android/
        └── chaquopy/
            ├── __init__.py
            └── chaquopy_bridge.py  # ← copied from android/chaquopy/
```

Symlink or copy `core/` into `app/src/main/python/` so Chaquopy can import it.

## Step 4: Build

```bash
./gradlew assembleDebug
```

Install:
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Step 5: Run Checks

1. **Cold start**: app copies assets, initializes Python, loads embeddings model (~10s on S25)
2. **Search**: type query → see sources populate → answer streams from MediaPipe
3. **Offline**: disable WiFi → search still works

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `lancedb` wheel not found for arm64 | Add `options "--no-binary"` in Chaquopy pip block; Chaquopy builds from source |
| `sentence-transformers` downloads model at runtime | Pre-download `multilingual-e5-small` to host, copy `~/.cache/torch/sentence_transformers/` into APK assets, reference by absolute path |
| APK > 100MB | Split: app base 20MB, models via Play Asset Delivery or direct HTTPS download on first launch |
| MediaPipe NPU not used | Check `adb logcat` for GPU delegate messages; fallback to CPU delegate is automatic |
| Python import errors | Verify `sys.path` includes repo root in `PythonAdapter.init()` |

## MediaPipe Model Conversion

If MediaPipe doesn't accept raw GGUF, convert via MediaPipe converter:

```bash
# Install converter
pip install mediapipe-model-maker

# Convert
python -m mediapipe_model_maker.llm.convert \
  --input_path gemma-3-4b-it-q4_k_m.gguf \
  --output_path gemma-3-4b-it.task \
  --backend gpu
```

Use `.task` file in `LlmInference.createFromFile()`.

## References
- Chaquopy docs: https://chaquo.com/chaquopy/doc/current/
- MediaPipe LLM Inference: https://developers.google.com/mediapipe/solutions/genai/llm_inference
- Play Feature Delivery: https://developer.android.com/guide/playcore/feature-delivery
