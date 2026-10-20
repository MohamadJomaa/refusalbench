"""Generation is crash-safe: a failing batch is retried one by one, a failing example is skipped and
retried later, and a second pass never duplicates finished work."""

from baseline.refusalbench_repro.runner import read_jsonl, run_generation

CFG = {"batch_size": 4, "provider": "fake", "model": "fake", "temperature": 1.0, "top_p": 1.0,
       "max_new_tokens": 8, "seed": 0, "precision": "4bit"}


def ex(i):
    return {"id": f"id{i}", "question": f"q{i}", "context": f"c{i}", "reference_answers": None,
            "expected_behavior": "REFUSE_OTHER", "uncertainty_type": "t", "intensity": "HIGH"}


class OomOnBatches:
    """Fails for any batch bigger than 1 (like a GPU out-of-memory), and always fails for example id3."""

    def generate_batch(self, prompts):
        if len(prompts) > 1 or "q3" in prompts[0]:
            raise RuntimeError("OutOfMemoryError")
        return ["ok"]


class Healthy:
    def generate_batch(self, prompts):
        return ["fine"] * len(prompts)


def test_out_of_memory_fallback_then_resume(tmp_path):
    out, err = tmp_path / "o.jsonl", tmp_path / "e.jsonl"
    examples = [ex(i) for i in range(8)]
    stats = run_generation(examples, OomOnBatches(), str(out), str(err), CFG, log=lambda m: None,
                           prompt_fn=lambda e: e["question"])
    assert sorted(r["example_id"] for r in read_jsonl(out)) == sorted(f"id{i}" for i in range(8) if i != 3)
    assert [r["example_id"] for r in read_jsonl(err)] == ["id3"]
    assert stats["generated"] == 7 and stats["failed"] == 1

    run_generation(examples, Healthy(), str(out), str(err), CFG, log=lambda m: None,
                   prompt_fn=lambda e: e["question"])
    ids = [r["example_id"] for r in read_jsonl(out)]
    assert len(ids) == 8 and len(set(ids)) == 8


def test_half_written_last_line_is_ignored(tmp_path):
    p = tmp_path / "o.jsonl"
    p.write_text('{"example_id": "a"}\n{"example_id": "b", "raw_resp', encoding="utf-8")
    assert [r["example_id"] for r in read_jsonl(p)] == ["a"]
