"""Model-level test of the scorer on a TINY RANDOM Qwen2 (CPU, a few seconds) with the REAL Qwen tokenizer.

It cannot say anything about Qwen's behavior. It checks the mechanics the real run depends on:
  * the refusal-code token prefixes are what we think they are,
  * left padding does not change a score (position ids are right),
  * cached teacher-forced scoring equals the slow cache-free reference,
  * hidden states have the right shape, the out-of-memory halving works.
Needs: torch, transformers, network access for the tokenizer files (a few MB).
Run from the repository root: python tests/ext2_tiny_model_check.py   (pytest runs it via tests/test_ext2_run.py)
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # tests/ -> the repository root
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "extension", "ext2_reading_signals"))

import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from scorer import WRAPPERS, HFScorer, refusal_variants, word_first_ids  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", padding_side="left")
enc = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731

# ---- token sets from the REAL tokenizer
variants = refusal_variants(enc)
for w, toks in variants.items():
    text = tok.decode(toks)
    print(f"  variant {w!r:12} -> {tok.convert_ids_to_tokens(toks)} = {text!r}")
    assert len(toks) >= 1 and text.startswith(w.strip()[:1] or "R"), (w, text)
    assert "REF" in (w + text), (w, text)  # the shared prefix reaches into the word REFUSE
yes, no = word_first_ids(enc, ["Yes", "yes", "YES"]), word_first_ids(enc, ["No", "no", "NO"])
assert yes and no and not set(yes) & set(no)
print("  Yes tokens", tok.convert_ids_to_tokens(yes), "| No tokens", tok.convert_ids_to_tokens(no))

# ---- a tiny random model
torch.manual_seed(0)
conf = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096)
model = Qwen2ForCausalLM(conf).eval()
sc = HFScorer({"s1_mode": "variants"}, log=lambda m: None, parts=(model, tok, torch))

ex = [{"id": f"e{i}", "question": "q" * (5 + 40 * i), "context": "some passage text " * (3 + 25 * i)} for i in range(4)]

# ---- the batch has very different lengths, so padding really happens
batch = sc.score(ex, "s1")
alone = [sc.score([e], "s1") for e in ex]
alone_scores = np.array([a["score"][0] for a in alone])
assert np.allclose(batch["score"], alone_scores, atol=1e-3), (batch["score"], alone_scores)
print("  left padding does not change S1:", np.round(batch["score"], 4))
b2 = sc.score(ex, "s2")
a2 = np.array([sc.score([e], "s2")["score"][0] for e in ex])
assert np.allclose(b2["score"], a2, atol=1e-3)
assert batch["hid_last"].shape == (4, 64) and batch["hid_mid"].shape == (4, 64)
assert np.isfinite(batch["score"]).all() and np.isfinite(b2["score"]).all()

# ---- cached teacher forcing == cache-free reference
from scorer import build_s1_prompt  # noqa: E402

prompts = [build_s1_prompt(e) for e in ex]
out, ids, mask, pos, lp0, _, _ = sc._forward(prompts, use_cache=True)
fast = sc._variant_logps_cache(out, mask, pos, lp0, len(prompts))
slow = sc._variant_logps_reforward(prompts)
assert np.abs(fast - slow).max() < 1e-3, np.abs(fast - slow).max()
print(f"  cached vs cache-free variant log-probs: max diff {np.abs(fast - slow).max():.2e}")
assert sc.preflight(ex) == "variants"

# ---- the first-token fallback runs too
sc.s1_mode = "first_token"
ft = sc.score(ex, "s1")
assert np.isfinite(ft["score"]).all()
sc.s1_mode = "variants"

# ---- out-of-memory halving: any batch above 1 "runs out of memory"
orig = sc._score_once


def flaky(examples, kind):
    if len(examples) > 1:
        raise torch.cuda.OutOfMemoryError("simulated")
    return orig(examples, kind)


sc._score_once = flaky
r = sc.score(ex, "s1")
assert np.allclose(r["score"], alone_scores, atol=1e-3) and len(r["score"]) == 4
print("  out-of-memory halving gives the same scores")
print("selftest_model: all assertions passed")
