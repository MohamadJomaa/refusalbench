"""
Run this once your team has a `.env` file and has picked a paper.

    python scripts/verify_setup.py            # fast, offline checks
    python scripts/verify_setup.py --online   # also downloads the dataset and makes one Gemini call

The first three checks are the course template's own, common to every team:
`.env` loads, an API key is set, and `pytest` passes. The checks after them
are specific to this project (RefusalBench-NQ with Qwen1.5-7B-Chat and a
Gemini judge), following the same `@check(...)` pattern.

Each check prints PASS, FAIL, or SKIP with a plain-language reason. Fix
FAILs before your first working session. A SKIP is fine: for example, no GPU
on a laptop only means the real runs happen on Colab.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # the checks read .env and data/ relative to the repository root

results = []
ONLINE = "--online" in sys.argv

# Common LLM provider key names. GEMINI_API_KEY is the name of our Colab
# secret; the judge accepts it or GOOGLE_API_KEY.
KNOWN_KEY_NAMES = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def check(name):
    """Wrap a check function so it prints PASS/FAIL/SKIP and records its
    result, instead of every check repeating that printing and error
    handling itself."""
    def decorator(fn):
        def wrapped():
            try:
                ok, detail = fn()
            except Exception as exc:  # noqa: BLE001, this is a diagnostic tool
                ok, detail = False, f"raised {type(exc).__name__}: {exc}"
            status = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
            results.append((name, status, detail))
            print(f"[{status:>4}] {name}: {detail}")
        return wrapped
    return decorator


# ------------------------------------------------------------ template checks
@check(".env file loads")
def check_dotenv():
    """Confirm a .env file exists in the repo root and load it, so every
    later check can read the keys it sets via os.getenv."""
    from dotenv import load_dotenv
    if not os.path.exists(".env"):
        return False, "no .env in the repo root. Copy .env.example to .env first."
    load_dotenv()
    return True, "found and loaded"


@check("at least one API key is set")
def check_api_key():
    """Confirm at least one of the common LLM provider keys is present.

    Does not confirm the key is valid, only that it is set. Run with
    --online to make one real call to the judge.
    """
    found = [name for name in KNOWN_KEY_NAMES if os.getenv(name)]
    if found:
        return True, f"found: {', '.join(found)}"
    return False, (
        "none of " + ", ".join(KNOWN_KEY_NAMES) + " is set in .env. "
        "This project needs GOOGLE_API_KEY for the Gemini judge (free: aistudio.google.com/apikey)."
    )


@check("pytest passes")
def check_pytest():
    """Run the repo's own test suite and report its last line of output,
    so a failure here is caught before your first working session."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True
    )
    ok = completed.returncode == 0
    lines = (completed.stdout + completed.stderr).strip().splitlines()
    tail = lines[-1] if lines else "no output"
    return ok, tail if ok else f"{tail} (run `python -m pytest -q` to see the whole error)"


# ------------------------------------------------------------ project checks
@check("Python 3.10 or newer")
def check_python():
    """The code uses list[dict] style type hints and other 3.10 features."""
    v = sys.version_info
    return (v >= (3, 10)), f"Python {v.major}.{v.minor}.{v.micro}"


@check("baseline and analysis packages import")
def check_packages():
    """The packages every run needs, with or without a GPU."""
    needed = {"datasets": "datasets", "yaml": "pyyaml", "numpy": "numpy", "google.genai": "google-genai",
              "sklearn": "scikit-learn", "scipy": "scipy", "matplotlib": "matplotlib"}
    missing = []
    for mod, pkg in needed.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        return False, f"missing: {', '.join(missing)}. Run: pip install -r requirements.txt"
    return True, "datasets, pyyaml, numpy, google-genai, scikit-learn, scipy, matplotlib"


@check("GPU stack for the real runs")
def check_gpu():
    """Qwen1.5-7B-Chat in 4-bit needs a CUDA GPU with about 6 GB free. Without
    one this is a SKIP, not a FAIL: the tests and --fake runs work on a CPU, and
    the real runs use the Colab notebooks."""
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError as exc:
        return None, f"{exc.name} not installed: fine on a laptop, the real runs happen on Colab"
    if not torch.cuda.is_available():
        return None, "no CUDA GPU here: fine for tests and --fake runs; use the Colab notebooks (T4) for real runs"
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False, "a GPU is present but bitsandbytes is missing (needed for 4-bit). pip install bitsandbytes"
    free, total = torch.cuda.mem_get_info()
    name = torch.cuda.get_device_name(0)
    ok = free / 2**30 >= 6
    return ok, f"{name}, {free / 2**30:.1f} of {total / 2**30:.1f} GB free" + ("" if ok else " (need about 6 GB)")


@check("the paper's prompt is unchanged")
def check_prompt():
    """The baseline prompt is copied verbatim from the paper's code. Its hash
    is recorded in every run folder, so an edit would break resuming and the
    faithfulness of the reproduction."""
    from baseline.refusalbench_repro.runner import prompt_hash
    h = prompt_hash()
    return h == "c0008c7504b3", f"hash {h}" + ("" if h == "c0008c7504b3" else " (expected c0008c7504b3)")


@check("fixed sample files in data/")
def check_data_files():
    """The 20-example smoke sample and the 800-example Extension 1 sample."""
    smoke = json.loads((ROOT / "data" / "smoke_sample_ids.json").read_text(encoding="utf-8"))
    nov = json.loads((ROOT / "data" / "novelty_sample_ids.json").read_text(encoding="utf-8"))
    ok = len(smoke["ids"]) == 20 and len(nov["ids"]) == 800 and len(nov["dev_pool_ids"]) == 800
    return ok, f"smoke {len(smoke['ids'])}, novelty {len(nov['ids'])} + dev pool {len(nov['dev_pool_ids'])}"


@check("committed baseline results are present")
def check_results():
    """The graded baseline run that the README and the report quote."""
    z = ROOT / "baseline" / "results" / "qwen15_7b_baseline_judged_day1.zip"
    if not z.exists():
        return False, f"missing {z.relative_to(ROOT)}"
    return True, f"{z.relative_to(ROOT)} ({z.stat().st_size / 1e6:.1f} MB)"


@check("RefusalBench-NQ downloads (--online)")
def check_dataset_online():
    """Downloads the benchmark from Hugging Face and checks its size."""
    if not ONLINE:
        return None, "skipped, run with --online to download the dataset"
    from baseline.refusalbench_repro.dataset_loader import load_refusalbench_nq
    ex = load_refusalbench_nq()
    n_ans = sum(e["is_answerable"] for e in ex)
    return (len(ex), n_ans) == (1600, 536), f"{len(ex)} examples, {n_ans} answerable (expected 1,600 and 536)"


@check("Gemini judge answers (--online)")
def check_judge_online():
    """One tiny call to the pinned judge model. Uses one request of the free daily quota."""
    if not ONLINE:
        return None, "skipped, run with --online to make one judge call"
    import yaml
    from baseline.refusalbench_repro.model_client import gemini_key
    from google import genai
    key = gemini_key()
    if not key:
        return False, "no GOOGLE_API_KEY or GEMINI_API_KEY set"
    model = yaml.safe_load((ROOT / "baseline" / "configs" / "qwen15_7b_baseline.yaml").read_text())["judge"]["model"]
    client = genai.Client(api_key=key)  # keep a reference: a client nobody holds is closed mid-call
    text = client.models.generate_content(model=model, contents="Reply with the word OK").text
    return True, f"{model} replied {(text or '').strip()[:20]!r}"


def main():
    """Run every check in order and exit non-zero if any failed, so this
    script works both read by a person and as a CI-style gate."""
    print("Setup check: the course template's three checks, then this project's own.")
    print("Offline by default; add --online to also download the dataset and call the judge once.\n")

    for fn in (check_dotenv, check_api_key, check_pytest, check_python, check_packages, check_gpu,
               check_prompt, check_data_files, check_results, check_dataset_online, check_judge_online):
        fn()

    failed = [r for r in results if r[1] == "FAIL"]
    print()
    if failed:
        print(f"{len(failed)} check(s) failed. Fix these before your first working session:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        sys.exit(1)
    print("All checks passed or were cleanly skipped.")


if __name__ == "__main__":
    main()
