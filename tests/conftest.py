"""Shared pytest fixtures. Nothing here touches a GPU, an API or the network."""

import importlib.util

import pytest

from tests._helpers import ROOT


@pytest.fixture
def synthetic_examples():
    from baseline.refusalbench_repro.schema_adapter import adapt_row
    from tests._helpers import synthetic_rows

    return [adapt_row(r) for r in synthetic_rows()]


@pytest.fixture
def run_all(monkeypatch, synthetic_examples):
    """baseline/scripts/run_all.py loaded as a module, with the dataset download replaced by synthetic rows."""
    path = ROOT / "baseline" / "scripts" / "run_all.py"
    spec = importlib.util.spec_from_file_location("run_all_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "load_refusalbench_nq", lambda: [dict(e) for e in synthetic_examples])

    def run(*args):
        monkeypatch.setattr("sys.argv", ["run_all.py", *args])
        mod.main()

    return run
