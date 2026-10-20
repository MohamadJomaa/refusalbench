"""The one test that must always pass, even before baseline work starts.

If this fails on a fresh checkout, fix your repository structure before
your first working session, not during it.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_expected_folders_exist():
    for name in ("baseline", "extension", "report", "data"):
        assert (ROOT / name).is_dir(), f"expected a {name}/ folder at the repo root"


def test_report_template_present():
    assert (ROOT / "report" / "REPORT.md").exists(), (
        "report/REPORT.md is missing, do not delete the template, fill it in instead"
    )
