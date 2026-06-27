"""Host build helper for the Engineer Companion Android APK.

Usage (from repo root, in .venv-win):
    python scripts/build_android.py [--skip-artifacts]

Steps performed:
  1. Validate prerequisites (JAVA_HOME, ANDROID_HOME/SDK, NDK, CMake).
  2. Regenerate host artifacts if needed:
       - scripts/build_android_index.py  -> flat binary index
       - scripts/gen_tokenizer_onnx.py   -> tokenizer.onnx
       - Check embedder ONNX model.onnx presence (heavy export NOT run automatically).
  3. Invoke Gradle: android/gradlew.bat :app:assembleDebug (streams output).
  4. On success, print APK path + size + install/run instructions.

Does NOT use buildozer or Kivy.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
ANDROID_DIR = REPO / "android"
SCRIPTS_DIR = REPO / "scripts"

APK_RELATIVE = Path("app/build/outputs/apk/debug/app-debug.apk")
APK_PATH = ANDROID_DIR / APK_RELATIVE

EMBEDDER_ONNX = (
    ANDROID_DIR
    / "app/src/main/assets/models/embedder_onnx/model.onnx"
)

# ---------------------------------------------------------------------------
# Resolve the .venv-win interpreter so host sub-scripts get the right deps.
# Fall back to sys.executable only if the venv python is absent.
# ---------------------------------------------------------------------------
_VENV_PYTHON = REPO / ".venv-win" / "Scripts" / "python.exe"
VENV_PYTHON: str = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

OPTIMUM_EXPORT_HINT = (
    "optimum-cli export onnx "
    '--model sentence-transformers/e5-small-v2 '
    '--task feature-extraction '
    "--optimize O2 "
    "android/app/src/main/assets/models/embedder_onnx/"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _warn(msg: str) -> None:
    print(f"  [!!]  {msg}", file=sys.stderr)


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", file=sys.stderr)


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Step 1 — prerequisite validation
# ---------------------------------------------------------------------------

def validate_prerequisites() -> bool:
    _section("Step 1: Validating prerequisites")
    ok = True

    # Java
    java_home = os.environ.get("JAVA_HOME", "")
    if java_home and Path(java_home).exists():
        javac = shutil.which("javac")
        if javac:
            ver = subprocess.run(
                ["javac", "-version"], capture_output=True, text=True
            )
            _ok(f"JAVA_HOME = {java_home}  ({ver.stderr.strip() or ver.stdout.strip()})")
        else:
            _ok(f"JAVA_HOME = {java_home} (javac not on PATH, but JAVA_HOME set)")
    else:
        _fail("JAVA_HOME is not set or does not exist.")
        _warn("Set JAVA_HOME to a JDK 17+ installation (JDK 21 preferred).")
        ok = False

    # Android SDK
    sdk_home = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT", "")
    if sdk_home and Path(sdk_home).exists():
        _ok(f"ANDROID_HOME = {sdk_home}")
    else:
        _fail("ANDROID_HOME (or ANDROID_SDK_ROOT) is not set or does not exist.")
        _warn("Install Android SDK via Android Studio SDK Manager and set ANDROID_HOME.")
        ok = False

    # NDK — look inside ANDROID_HOME/ndk or ANDROID_HOME/ndk-bundle
    ndk_found = False
    if sdk_home:
        ndk_dirs = list(Path(sdk_home, "ndk").glob("*")) if Path(sdk_home, "ndk").exists() else []
        ndk_bundle = Path(sdk_home, "ndk-bundle")
        if ndk_dirs:
            _ok(f"NDK found: {[str(d) for d in ndk_dirs]}")
            ndk_found = True
        elif ndk_bundle.exists():
            _ok(f"NDK bundle found: {ndk_bundle}")
            ndk_found = True
    if not ndk_found:
        _fail("Android NDK not found inside ANDROID_HOME/ndk/.")
        _warn("Install NDK 28.2.x via SDK Manager: SDK Tools -> NDK (Side by side).")
        ok = False

    # CMake — check PATH first, then inside SDK
    cmake = shutil.which("cmake")
    if cmake:
        ver = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
        first_line = ver.stdout.splitlines()[0] if ver.stdout else "unknown"
        _ok(f"cmake found: {cmake}  ({first_line})")
    elif sdk_home:
        cmake_candidates = list(Path(sdk_home, "cmake").glob("*/bin/cmake.exe"))
        if cmake_candidates:
            _ok(f"CMake found in SDK: {cmake_candidates[0]}")
        else:
            _fail("CMake not found on PATH or in ANDROID_HOME/cmake/.")
            _warn("Install via SDK Manager: SDK Tools -> CMake 3.22+, or add to PATH.")
            ok = False
    else:
        _fail("CMake not found on PATH.")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Step 2 — host artifact regeneration
# ---------------------------------------------------------------------------

def regenerate_artifacts(skip: bool = False) -> bool:
    _section("Step 2: Host artifacts")
    if skip:
        print("  Skipped (--skip-artifacts).")
        return True

    ok = True

    # --- flat binary index ---
    index_script = SCRIPTS_DIR / "build_android_index.py"
    if index_script.exists():
        print(f"  Running {index_script.name} (python: {VENV_PYTHON}) ...")
        result = subprocess.run(
            [VENV_PYTHON, str(index_script)],
            cwd=str(REPO),
        )
        if result.returncode == 0:
            _ok("build_android_index.py completed.")
        else:
            _fail(f"build_android_index.py exited with code {result.returncode}.")
            ok = False
    else:
        _warn(f"Index script not found: {index_script}  (skipping).")

    # --- tokenizer.onnx ---
    tok_script = SCRIPTS_DIR / "gen_tokenizer_onnx.py"
    if tok_script.exists():
        print(f"  Running {tok_script.name} (python: {VENV_PYTHON}) ...")
        result = subprocess.run(
            [VENV_PYTHON, str(tok_script)],
            cwd=str(REPO),
        )
        if result.returncode == 0:
            _ok("gen_tokenizer_onnx.py completed.")
        else:
            _fail(f"gen_tokenizer_onnx.py exited with code {result.returncode}.")
            ok = False
    else:
        _warn(f"Tokenizer script not found: {tok_script}  (skipping).")

    # --- embedder ONNX model.onnx (heavy — NOT auto-run) ---
    # model.onnx must be the FP16 version (~224 MB, down from 448 MB FP32).
    # To regenerate it from scratch run (once, takes ~1-2 min):
    #   python scripts/quantize_embedder_onnx.py
    # That script converts FP32→FP16, patches Cast nodes, and writes
    # model_fp16_fixed.onnx. Then adopt it with:
    #   Copy-Item .../model_fp16_fixed.onnx .../model.onnx -Force
    if EMBEDDER_ONNX.exists():
        size_mb = EMBEDDER_ONNX.stat().st_size / (1024 ** 2)
        _ok(f"Embedder ONNX present: {EMBEDDER_ONNX}  ({size_mb:.0f} MB)")
    else:
        _fail(f"Embedder ONNX NOT found: {EMBEDDER_ONNX}")
        print()
        print("  This file is required for on-device retrieval.")
        print("  Run the following command (once) to export it (takes ~5-10 min):")
        print()
        print(f"      {OPTIMUM_EXPORT_HINT}")
        print()
        print("  (The export is NOT triggered automatically because it is slow and")
        print("   requires optimum[exporters] + transformers installed.)")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Step 3 — Gradle build
# ---------------------------------------------------------------------------

def run_gradle_build() -> bool:
    _section("Step 3: Gradle assembleDebug")
    gradlew = ANDROID_DIR / "gradlew.bat"
    if not gradlew.exists():
        _fail(f"Gradle wrapper not found: {gradlew}")
        return False

    cmd = [str(gradlew), ":app:assembleDebug"]
    print(f"  Running: {' '.join(cmd)}")
    print(f"  Working dir: {ANDROID_DIR}")
    print()

    # Stream output in real time
    result = subprocess.run(cmd, cwd=str(ANDROID_DIR))

    if result.returncode != 0:
        _fail(f"Gradle build failed (exit code {result.returncode}).")
        return False

    _ok("Gradle build succeeded.")
    return True


# ---------------------------------------------------------------------------
# Step 4 — success report
# ---------------------------------------------------------------------------

def print_success_report() -> None:
    _section("Step 4: Build complete")
    if APK_PATH.exists():
        size_mb = APK_PATH.stat().st_size / (1024 ** 2)
        print(f"  APK:  {APK_PATH}")
        print(f"  Size: {size_mb:.1f} MB")
    else:
        _warn(f"Expected APK not found at {APK_PATH}")

    print()
    print("  ── On-device install / run ─────────────────────────────────")
    print()
    print("  1. Install APK:")
    print(r"     adb install -r app\build\outputs\apk\debug\app-debug.apk")
    print()
    print("  2. Push GGUF model (2.4 GB) — two options:")
    print(r"     a) Push to /sdcard/Download/ and import via the in-app SetupScreen:")
    print(r"        adb push assets\models\gemma-3-4b-it-Q4_K_M.gguf /sdcard/Download/")
    print(r"     b) Push directly to the app files dir (needs adb root or run-as):")
    print(r"        adb push assets\models\gemma-3-4b-it-Q4_K_M.gguf")
    print(r"            /data/data/com.varian.engcomp/files/models/gemma-3-4b-it-Q4_K_M.gguf")
    print()
    print("  3. Launch the app and use the SetupScreen to select the GGUF if pushed to /sdcard/.")
    print()
    print("  4. Monitor logcat:")
    print(r'     adb logcat | findstr "engineer_companion EngineerCompanion ort llama"')
    print("     Look for: UnsatisfiedLinkError, model load errors, tokenizer issues.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Host build helper for Engineer Companion Android APK."
    )
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Skip host artifact regeneration (index + tokenizer).",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Validate and regenerate artifacts only; skip Gradle build.",
    )
    args = parser.parse_args()

    prereqs_ok = validate_prerequisites()
    if not prereqs_ok:
        print(
            "\n  Prerequisites failed. Fix the errors above before building.\n",
            file=sys.stderr,
        )
        return 1

    artifacts_ok = regenerate_artifacts(skip=args.skip_artifacts)
    if not artifacts_ok:
        print(
            "\n  One or more host artifacts are missing. Fix above, then re-run.\n",
            file=sys.stderr,
        )
        return 1

    if args.no_build:
        print("\n  --no-build: skipping Gradle step.\n")
        return 0

    build_ok = run_gradle_build()
    if not build_ok:
        return 1

    print_success_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
