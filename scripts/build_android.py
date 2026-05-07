"""Build script: Android APK via Kivy + python-for-android (buildozer).

Usage:
    python scripts/build_android.py

Requires: buildozer, android-ndk, android-sdk, kivy
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BUILDozer_CONF = """
[app]
title = Engineer Companion
package.name = engineercompanion
package.domain = org.engcomp
source.dir = .
source.include_exts = py,data,txt,kv,json
version = 0.1.0
requirements = python3,kivy,numpy,structlog
orientation = all
fullscreen = 0
android.arch = arm64-v8a
android.api = 33
android.minapi = 24
android.ndk = 25c
android.sdk = 33

[buildozer]
log_level = 2
warn_on_root = 1
"""


def write_buildozer() -> None:
    path = REPO / "buildozer.spec"
    if not path.exists():
        path.write_text(BUILDozer_CONF)
        print(f"Created {path}")


def build() -> None:
    write_buildozer()
    # For now this is a placeholder: actual build requires Android SDK/NDK setup
    cmd = ["buildozer", "-v", "android", "debug"]
    print(f"Running: {' '.join(cmd)}")
    print("NOTE: Ensure ANDROIDSDK / ANDROIDNDK / JAVA_HOME env vars are set.")
    subprocess.run(cmd, cwd=REPO)


if __name__ == "__main__":
    build()
