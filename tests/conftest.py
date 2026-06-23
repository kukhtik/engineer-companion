"""
Pytest session configuration for engineer-companion.

Windows DLL-ordering fix
------------------------
In a pytest-qt session the load order QtTest + torch (libiomp5md.dll) +
sklearn→pandas→pyarrow + lancedb.connect() triggers a Windows access
violation (exit code -1073741819 / 0xC0000005).

Root cause: pyarrow's Arrow C++ runtime and torch's libiomp5md.dll both
register OpenMP threading; if pyarrow is imported *after* torch has already
locked the OMP thread pool the second initialisation races and faults.

Fix: import pyarrow and lancedb here, at conftest import time, *before* any
test module (and therefore before pytest-qt loads QtTest or any test imports
torch).  This conftest is loaded first in the collection phase, so the Arrow
C++ runtime is already resident when torch's DLLs arrive.
"""

import os

# Offscreen Qt platform — must be set before any PySide6/Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Pre-warm pyarrow/Arrow C++ runtime before torch's libiomp5md.dll and Qt
# threading get loaded, to avoid a Windows OpenMP DLL-ordering access violation.
try:
    import pyarrow          # noqa: F401
    import pyarrow.dataset  # noqa: F401
    import lancedb          # noqa: F401
except Exception:
    pass
