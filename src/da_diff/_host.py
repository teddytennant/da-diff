"""Preload C++/zlib on hosts (NixOS) that hide them from the dynamic loader."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

_PRELOAD = ("libstdc++.so.6", "libgcc_s.so.1", "libz.so.1")
_done = False


def preload() -> None:
    global _done
    if _done:
        return
    _done = True
    roots = []
    for env in ("NIX_LD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        roots.extend(p for p in os.environ.get(env, "").split(":") if p)
    extra = Path("/run/current-system/sw/share/nix-ld/lib")
    if extra.is_dir():
        roots.append(str(extra))
    seen = set()
    for root in roots:
        for name in _PRELOAD:
            cand = Path(root) / name
            key = str(cand)
            if key in seen or not cand.exists():
                continue
            seen.add(key)
            try:
                ctypes.CDLL(key, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
