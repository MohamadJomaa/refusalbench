"""
Run this once your team has a `.env` file and has picked a paper.

    python scripts/verify_setup.py

This checks only what every team needs regardless of topic or paper:
that `.env` loads, that at least one API key is set, and that `pytest`
passes. It deliberately does not check anything baseline-specific (a
vector store, a memory service, a particular model) since that varies by
paper; add your own checks here once you know what your baseline
actually needs, following the same `@check(...)` pattern below.

Each check prints PASS, FAIL, or SKIP with a plain-language reason. Fix
FAILs before your first working session.
"""

import os
import subprocess
import sys

results = []

# Common LLM provider key names. Add your baseline's own key name here if
# it uses something not listed, rather than renaming it to match this list.
KNOWN_KEY_NAMES = ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


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

    Does not confirm the key is valid, only that it is set. Your chosen
    baseline may need a key not in KNOWN_KEY_NAMES above, in that case
    add it to the tuple rather than treating this check as sufficient.
    """
    found = [name for name in KNOWN_KEY_NAMES if os.getenv(name)]
    if found:
        return True, f"found: {', '.join(found)}"
    return False, (
        "none of " + ", ".join(KNOWN_KEY_NAMES) + " is set in .env. "
        "Add whichever your chosen baseline actually needs."
    )


@check("pytest passes")
def check_pytest():
    """Run the repo's own test suite and report its last line of output,
    so a failure here is caught before your first working session."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True
    )
    ok = completed.returncode == 0
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "no output"
    return ok, tail


def main():
    """Run every check in order and exit non-zero if any failed, so this
    script works both read by a person and as a CI-style gate."""
    print("Running the base setup check, common to every team regardless of")
    print("topic or paper. Add your own baseline-specific checks below this")
    print("script's existing ones once you know what your paper needs.\n")

    check_dotenv()
    check_api_key()
    check_pytest()

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
