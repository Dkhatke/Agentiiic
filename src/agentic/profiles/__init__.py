import tomllib
from pathlib import Path

from ..loop import Profile

_DIR = Path(__file__).parent


def load_profile(name: str) -> Profile:
    path = _DIR / f"{name}.toml"
    if not path.exists():
        available = ", ".join(p.stem for p in _DIR.glob("*.toml"))
        raise SystemExit(f"unknown profile '{name}' (available: {available})")
    data = tomllib.loads(path.read_text())
    data["keywords"] = tuple(data.get("keywords", ()))
    data["checks"] = tuple(data.get("checks", ()))
    return Profile(**data)


def all_profiles() -> list[Profile]:
    return [load_profile(p.stem) for p in sorted(_DIR.glob("*.toml"))]
