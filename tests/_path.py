"""Make ``src/`` importable when running ``python -m unittest discover -s tests``."""
import importlib.util
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


HAS_TORCH = has("torch")
HAS_CAUSALLEARN = has("causallearn")
HAS_SEABORN = has("seaborn")
