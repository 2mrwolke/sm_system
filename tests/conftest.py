"""Pytest configuration.

The project uses a ``src/`` layout. Adding ``src`` to ``sys.path`` ensures tests
can be run directly from a source checkout without requiring an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path


SRC = (Path(__file__).resolve().parents[1] / "src").as_posix()
if SRC not in sys.path:
    sys.path.insert(0, SRC)
