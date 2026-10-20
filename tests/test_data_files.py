"""The two fixed sample files in data/ have the sizes and properties documented in data/README.md."""

import json
import re
from collections import Counter

from tests._helpers import ROOT

ID_RE = re.compile(r"RB-NQ_([a-z0-9]+)_(-?\d+)_(P-[A-Za-z]+)_(LOW|MEDIUM|HIGH)_")


def _load(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def test_smoke_sample():
    d = _load("smoke_sample_ids.json")
    assert d["seed"] == 0 and d["n"] == 20 == len(set(d["ids"]))


def test_novelty_sample_and_dev_pool():
    d = _load("novelty_sample_ids.json")
    ids, dev = d["ids"], d["dev_pool_ids"]
    assert len(ids) == len(set(ids)) == 800 and len(dev) == len(set(dev)) == 800
    assert not set(ids) & set(dev), "the dev pool must not overlap the evaluation sample"
    parsed = [ID_RE.match(i).groups() for i in ids]
    assert Counter(p[3] for p in parsed) == {"LOW": 268, "MEDIUM": 265, "HIGH": 267}
    assert len({p[1] for p in parsed}) == 100, "every source question must be in the sample"
