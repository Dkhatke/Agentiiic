import tomllib
from pathlib import Path

from ..loop import Profile

_DIR = Path(__file__).parent


def load_profile(name: str) -> Profile:
    path = _DIR / f"{name}.toml"
    if not path.exists():
        available = ", ".join(p.stem for p in _DIR.glob("*.toml"))
        raise SystemExit(f"unknown profile '{name}' (available: {available})")
    return Profile(**tomllib.loads(path.read_text()))
