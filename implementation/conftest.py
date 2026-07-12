"""Make `src/` importable in tests regardless of how pytest is launched."""

import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
