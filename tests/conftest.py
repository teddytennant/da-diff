"""Load host C++/zlib before test modules import jax."""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from da_diff._host import preload

preload()
