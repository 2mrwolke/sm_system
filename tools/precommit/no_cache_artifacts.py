#!/usr/bin/env python3
"""Fail if cache / bytecode artifacts are staged/tracked.

This prevents committing Python bytecode and tool caches that create noisy diffs and
break reproducible packaging.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Iterable


DISALLOWED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(^|/)(__pycache__)(/|$)"),
    re.compile(r"\.pyc$"),
    re.compile(r"(^|/)\.pytest_cache(/|$)"),
    re.compile(r"(^|/)\.mypy_cache(/|$)"),
    re.compile(r"(^|/)\.ruff_cache(/|$)"),
    re.compile(r"(^|/)\.ipynb_checkpoints(/|$)"),
    re.compile(r"(^|/)build(/|$)"),
    re.compile(r"(^|/)dist(/|$)"),
    re.compile(r"\.egg-info(/|$)"),
]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip("\n")


def _matches_any(path: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(p.search(path) for p in patterns)


def main() -> int:
    try:
        # Only enforce on staged changes if running in a commit context.
        staged = _git("diff", "--cached", "--name-only").splitlines()
        if staged and any(_matches_any(p, DISALLOWED_PATTERNS) for p in staged):
            bad = [p for p in staged if _matches_any(p, DISALLOWED_PATTERNS)]
            print("Disallowed cache/bytecode artifacts are staged:")
            print("\n".join(f"  - {p}" for p in bad))
            print("\nRemove them (or add to .gitignore) before committing.")
            return 1

        # Also guard against already-tracked artifacts.
        tracked = _git("ls-files").splitlines()
        bad_tracked = [p for p in tracked if _matches_any(p, DISALLOWED_PATTERNS)]
        if bad_tracked:
            print("Disallowed cache/bytecode artifacts are tracked in git:")
            print("\n".join(f"  - {p}" for p in bad_tracked))
            return 1

        return 0
    except subprocess.CalledProcessError as e:
        print("Failed to run git command:", e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
