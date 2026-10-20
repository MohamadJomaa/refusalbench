"""Prompt and reply handling per experimental condition.

The baseline lives here. Every other condition belongs to an extension folder
(`extension/<name>/conditions.py`, or `extension/<name>/code/conditions.py`) and is loaded as a plug-in when
the run config names that folder with `experiment_dir`. An experiment module must define
    build_prompt_for(condition, example) -> str
    extract_effective(condition, raw)    -> {"effective": str, "state": str|None, "flags": list}
Without a plug-in only "baseline" works, so the baseline pipeline never depends on any experiment's code.
"""

import importlib.util
import os

from .prompt_builder import build_prompt

_plugin = None


def load_experiment(folder: str, repo_root: str) -> None:
    """Load <folder>/conditions.py (or <folder>/code/conditions.py) as the plug-in for non-baseline conditions."""
    global _plugin
    base = folder if os.path.isabs(folder) else os.path.join(repo_root, folder)
    candidates = [os.path.join(base, "conditions.py"), os.path.join(base, "code", "conditions.py")]
    path = next((c for c in candidates if os.path.exists(c)), None)
    if path is None:
        raise FileNotFoundError(f"experiment_dir has no conditions.py: {candidates}")
    spec = importlib.util.spec_from_file_location("experiment_conditions", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _plugin = mod


def _need_plugin(condition: str):
    if _plugin is None:
        raise RuntimeError(f"condition {condition!r} needs an experiment plug-in: set `experiment_dir` in the "
                           "config (for example extension/ext1_diagnose_first)")
    return _plugin


def build_prompt_for(condition: str, example: dict) -> str:
    if condition == "baseline":
        return build_prompt(example)
    return _need_plugin(condition).build_prompt_for(condition, example)


def extract_effective(condition: str, raw: str) -> dict:
    """What gets scored. `raw` is always preserved elsewhere."""
    if condition == "baseline":
        return {"effective": raw or "", "state": None, "flags": []}
    return _need_plugin(condition).extract_effective(condition, raw)
