"""Load prompt templates from YAML files in the prompts/ directory."""

from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
_cache: dict[str, dict] = {}


def load_prompt(name: str) -> dict:
    """Load a prompt YAML file by path relative to prompts/.

    Example: load_prompt("agents/methodology_critic")
    Returns: {"version": "1.0", "description": "...", "system": "...", ...}
    """
    if name in _cache:
        return _cache[name]

    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    _cache[name] = data
    return data
