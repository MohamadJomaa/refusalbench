"""Extension 2 scorers. One forward pass per example, NO text generation.

Two prompts are scored per example:
  S1  the original baseline prompt. Score = log-odds that Qwen's reply starts a refusal code.
  S2  a yes/no sufficiency question.  Score = log-odds of "No" against "Yes".
For both prompts the last-token hidden states are saved too (used by the S3 linear probe in analysis.py).

Higher score always means "refuse".

How S1 reads the refusal probability without trusting Qwen's text: a refusal code is several tokens
(`RE`, `FUSE`, ...), and Qwen sometimes wraps it (`**REFUSE...`, `` `REFUSE...``, `ANSWER: REFUSE...`).
For each wrapper we take the longest token prefix shared by all seven codes and compute its probability by
teacher forcing; P(refuse) is the sum over wrappers. The wrapper forms are mutually exclusive openings.
"""

import hashlib
import os

import numpy as np

from baseline.refusalbench_repro.prompt_builder import PROMPT_TEMPLATE, build_prompt
from baseline.refusalbench_repro.schema_adapter import REFUSAL_CODES

WRAPPERS = ["", "**", "`", "ANSWER: ", "Answer: "]

SUFFICIENCY_TEMPLATE = """
        Read the QUERY and the CONTEXT PASSAGES carefully.

        QUERY:
        {query}

        CONTEXT PASSAGES:
        {context}

        QUESTION: Can the QUERY be answered completely and faithfully using only the CONTEXT PASSAGES? Answer No if the query or context is ambiguous, if the context contradicts itself, if the needed information is missing, if the query rests on a false premise, if the information is at the wrong level of detail, or if the query asks for an opinion or a prediction. Otherwise answer Yes.

        Answer with exactly one word: Yes or No.
        """

YES_WORDS = ["Yes", "yes", "YES"]
NO_WORDS = ["No", "no", "NO"]


def build_s1_prompt(example: dict) -> str:
    return build_prompt(example)


def build_s2_prompt(example: dict) -> str:
    return SUFFICIENCY_TEMPLATE.format(query=example["question"], context=example["context"])


def prompts_hash() -> str:
    h = hashlib.sha256((PROMPT_TEMPLATE + SUFFICIENCY_TEMPLATE + repr(WRAPPERS)).encode("utf-8"))
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ token sets (pure functions, unit-tested)
def common_prefix(seqs):
    """Longest shared prefix of several token-id lists."""
    out = []
    for toks in zip(*seqs):
        if len(set(toks)) != 1:
            break
        out.append(toks[0])
    return out


def refusal_variants(encode):
    """encode(str) -> list[int] without special tokens. Returns {wrapper: shared token prefix of the 7 codes}."""
    variants = {}
    for w in WRAPPERS:
        variants[w] = common_prefix([encode(w + code) for code in REFUSAL_CODES])
    return variants


def word_first_ids(encode, words):
    ids = []
    for w in words:
        t = encode(w)
        if t:
            ids.append(t[0])
    return sorted(set(ids))


def _logsumexp(a):
    a = np.asarray(a, dtype=np.float64)
    m = a.max()
    return float(m + np.log(np.exp(a - m).sum()))


def log_odds_from_logprobs(logp_pos: float, eps: float = 1e-12) -> float:
    """log(p / (1 - p)) from log p, without overflow."""
    p = min(float(np.exp(logp_pos)), 1.0 - eps)
    return float(np.log(max(p, eps)) - np.log1p(-p))


def s1_from_variant_logprobs(variant_logps):
    """variant_logps: list of log P(variant). Sum the probabilities of the exclusive openings."""
    logp = _logsumexp(variant_logps)
    return logp, log_odds_from_logprobs(logp)


def s2_from_logprobs(lp_vocab_no, lp_vocab_yes):
    """Both args: lists of log-probs of the No tokens and of the Yes tokens. Returns (score, mass)."""
    lno, lyes = _logsumexp(lp_vocab_no), _logsumexp(lp_vocab_yes)
    return lno - lyes, float(np.exp(lno) + np.exp(lyes))


# ------------------------------------------------------------------ the real scorer (Hugging Face)
class HFScorer:
    kinds = ("s1", "s2")

    def __init__(self, cfg: dict, log=print, parts=None):
        """`parts=(model, tokenizer, torch)` lets the tests pass a tiny model instead of loading the 7B weights."""
        self.log = log
        if parts is not None:
            self.model, self.tok, self.torch = parts
        else:
            from baseline.refusalbench_repro.model_client import HFClient

            self.client = HFClient(cfg)  # loads tokenizer (left padding) and the 4-bit model, seeds everything
            self.torch, self.tok, self.model = self.client.torch, self.client.tok, self.client.model
        enc = lambda s: self.tok(s, add_special_tokens=False).input_ids  # noqa: E731
        self.variants = refusal_variants(enc)
        self.yes_ids = word_first_ids(enc, YES_WORDS)
        self.no_ids = word_first_ids(enc, NO_WORDS)
        for w, toks in self.variants.items():
            self.log(f"  S1 variant {w!r:12} -> tokens {self.tok.convert_ids_to_tokens(toks)}")
        self.log(f"  S2 Yes ids {self.tok.convert_ids_to_tokens(self.yes_ids)}, "
                 f"No ids {self.tok.convert_ids_to_tokens(self.no_ids)}")
        self.s1_mode = cfg.get("s1_mode", "variants")  # "variants" (cache) or "first_token" (fallback)

    # -- model call
    def _encode(self, prompts):
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        mask = enc["attention_mask"]
        pos = (mask.cumsum(-1) - 1).clamp(min=0)  # left padding: real tokens must start at position 0
        return enc["input_ids"], mask, pos

    def _base(self):
        return self.model.model  # Qwen2Model: embeddings + layers + final norm (lm_head is applied by hand)

    def _forward(self, prompts, use_cache):
        torch = self.torch
        ids, mask, pos = self._encode(prompts)
        with torch.no_grad():
            out = self._base()(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=use_cache,
                               output_hidden_states=True)
            hs = out.hidden_states
            last = out.last_hidden_state[:, -1]
            mid = hs[len(hs) // 2][:, -1]
            logits = self.model.lm_head(last).float()
            lp = torch.log_softmax(logits, dim=-1)
        return out, ids, mask, pos, lp, last, mid

    def _variant_logps_cache(self, out, mask, pos, lp0, B):
        """log P(variant) for every wrapper, reusing the prompt's cache. Cache is cropped back after each."""
        torch = self.torch
        res = np.zeros((B, len(self.variants)), dtype=np.float64)
        for vi, (w, toks) in enumerate(self.variants.items()):
            lp = lp0[:, toks[0]].double().cpu().numpy()
            if len(toks) > 1:
                feed = torch.tensor(toks[:-1], device=mask.device).unsqueeze(0).expand(B, -1)
                m = len(toks) - 1
                m2 = torch.cat([mask, torch.ones(B, m, dtype=mask.dtype, device=mask.device)], dim=1)
                p2 = pos[:, -1:] + torch.arange(1, m + 1, device=pos.device).unsqueeze(0)
                with torch.no_grad():
                    o2 = self._base()(input_ids=feed, attention_mask=m2, position_ids=p2,
                                      past_key_values=out.past_key_values, use_cache=True)
                    lg = self.model.lm_head(o2.last_hidden_state).float()
                    lpj = torch.log_softmax(lg, dim=-1)
                for j in range(1, len(toks)):
                    lp = lp + lpj[:, j - 1, toks[j]].double().cpu().numpy()
                out.past_key_values.crop(-m)  # drop the m tokens just fed (negative form works in old and new versions)
            res[:, vi] = lp
        return res

    def _variant_logps_reforward(self, prompts):
        """Slow, cache-free reference used by the pre-flight check: re-run prompt + variant tokens in full."""
        torch = self.torch
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True) for p in prompts]
        base_ids = [self.tok(t, add_special_tokens=False).input_ids for t in texts]
        res = np.zeros((len(prompts), len(self.variants)))
        for vi, toks in enumerate(self.variants.values()):
            for bi, b in enumerate(base_ids):
                ids = torch.tensor([b + toks], device=self.model.device)
                with torch.no_grad():
                    h = self._base()(input_ids=ids).last_hidden_state[0]
                    lp = torch.log_softmax(self.model.lm_head(h).float(), -1)
                s = 0.0
                for j, t in enumerate(toks):
                    s += float(lp[len(b) - 1 + j, t])
                res[bi, vi] = s
        return res

    def preflight(self, examples):
        """Check on a few real examples that the cached variant scoring equals the cache-free reference.
        If not (or if it crashes), switch to the first-token fallback and say so."""
        prompts = [build_s1_prompt(e) for e in examples[:3]]
        try:
            out, ids, mask, pos, lp0, _, _ = self._forward(prompts, use_cache=True)
            fast = self._variant_logps_cache(out, mask, pos, lp0, len(prompts))
            slow = self._variant_logps_reforward(prompts)
            err = float(np.abs(fast - slow).max())
            self.log(f"  pre-flight: cached vs full re-forward variant log-probs differ by at most {err:.3f}")
            if err > 0.5:
                raise RuntimeError(f"cache path disagrees with the reference (max diff {err:.2f})")
            self.s1_mode = "variants"
        except Exception as e:  # noqa: BLE001 - any failure means: use the simple, robust fallback
            self.log(f"  !!! pre-flight failed ({type(e).__name__}: {str(e)[:200]}). "
                     "Falling back to S1 = first token only (documented in the report).")
            self.s1_mode = "first_token"
        self.free()
        return self.s1_mode

    def free(self):
        import gc

        gc.collect()
        self.torch.cuda.empty_cache()

    # -- scoring
    def _score_once(self, examples, kind):
        prompts = [(build_s1_prompt if kind == "s1" else build_s2_prompt)(e) for e in examples]
        B = len(prompts)
        out, ids, mask, pos, lp0, last, mid = self._forward(prompts, use_cache=(kind == "s1" and self.s1_mode == "variants"))
        top1 = self.tok.convert_ids_to_tokens(lp0.argmax(-1).tolist())
        if kind == "s1":
            if self.s1_mode == "variants":
                vl = self._variant_logps_cache(out, mask, pos, lp0, B)
            else:
                first = [toks[0] for toks in self.variants.values()]
                vl = lp0[:, first].double().cpu().numpy()
            pairs = [s1_from_variant_logprobs(vl[i]) for i in range(B)]
            logp = np.array([p[0] for p in pairs]); score = np.array([p[1] for p in pairs]); mass = np.exp(logp)
        else:
            lno = lp0[:, self.no_ids].double().cpu().numpy(); lyes = lp0[:, self.yes_ids].double().cpu().numpy()
            pairs = [s2_from_logprobs(lno[i], lyes[i]) for i in range(B)]
            score = np.array([p[0] for p in pairs]); mass = np.array([p[1] for p in pairs])
        return {"score": score, "mass": mass, "top1": np.array(top1),
                "hid_last": last.half().cpu().numpy(), "hid_mid": mid.half().cpu().numpy()}

    def score(self, examples, kind):
        """Halves the batch on CUDA out-of-memory. The retry is outside the except block so memory can be freed."""
        try:
            return self._score_once(examples, kind)
        except self.torch.cuda.OutOfMemoryError:
            pass
        self.free()
        if len(examples) == 1:
            return self._score_once(examples, kind)
        mid = len(examples) // 2
        a, b = self.score(examples[:mid], kind), self.score(examples[mid:], kind)
        return {k: np.concatenate([a[k], b[k]]) for k in a}


# ------------------------------------------------------------------ fake scorer (tests only)
class FakeScorer:
    """Synthetic scores that correlate with the true label. TEST ONLY: never used for a real result."""

    kinds = ("s1", "s2")
    s1_mode = "variants"

    def __init__(self, cfg: dict, log=print):
        self.cfg = cfg

    def preflight(self, examples):
        return self.s1_mode

    def score(self, examples, kind):
        n = len(examples)
        score, mass, top1 = np.zeros(n), np.ones(n), np.array(["RE"] * n)
        H = 16
        hl, hm = np.zeros((n, H), np.float16), np.zeros((n, H), np.float16)
        for i, e in enumerate(examples):
            rng = np.random.default_rng(int(hashlib.sha256((kind + e["id"]).encode()).hexdigest()[:8], 16))
            refuse = 0.0 if e["is_answerable"] else 1.0
            score[i] = (-1.0 + 2.0 * refuse) * (1.0 if kind == "s1" else 0.6) + rng.normal(0, 1.5)
            v = rng.normal(0, 1, H); v[0] += 2.0 * refuse
            hl[i], hm[i] = v, v[::-1]
        return {"score": score, "mass": mass, "top1": top1, "hid_last": hl, "hid_mid": hm}


# ------------------------------------------------------------------ resumable shards
def shard_path(shard_dir, kind, start):
    return os.path.join(shard_dir, f"{kind}_{start:05d}.npz")


def done_ids(shard_dir, kind):
    done = set()
    if os.path.isdir(shard_dir):
        for fn in os.listdir(shard_dir):
            if fn.startswith(kind + "_") and fn.endswith(".npz"):
                try:
                    done.update(np.load(os.path.join(shard_dir, fn), allow_pickle=False)["ids"].tolist())
                except Exception:  # a half-written file from a crash: that shard is simply redone
                    pass
    return done


def save_shard(shard_dir, kind, start, ids, res):
    os.makedirs(shard_dir, exist_ok=True)
    final = shard_path(shard_dir, kind, start)
    tmp = final + ".tmp.npz"
    np.savez_compressed(tmp, ids=np.array(ids), **res)
    os.replace(tmp, final)


def run_scoring(examples, scorer, shard_dir, kind, shard_size, batch_size, log=print):
    """Score every example not yet in a shard. Shards are written atomically, so a crash loses one shard at most."""
    import time

    done = done_ids(shard_dir, kind)
    todo = [e for e in examples if e["id"] not in done]
    log(f"--- {kind}: {len(examples)} examples, {len(todo)} still to score")
    t0, n = time.time(), 0
    for s in range(0, len(todo), shard_size):
        chunk = todo[s:s + shard_size]
        parts = []
        for b in range(0, len(chunk), batch_size):
            parts.append(scorer.score(chunk[b:b + batch_size], kind))
        res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        save_shard(shard_dir, kind, len(done) + s, [e["id"] for e in chunk], res)
        n += len(chunk)
        el = time.time() - t0
        log(f"  {kind}: {n}/{len(todo)} this call ({el / n:.2f}s per example, about {el / n * (len(todo) - n) / 60:.0f} min left)")
    return {"scored": n, "seconds": time.time() - t0}


def load_shards(shard_dir, kind):
    """Returns {id: {field: value}} for one kind."""
    out = {}
    for fn in sorted(os.listdir(shard_dir)):
        if fn.startswith(kind + "_") and fn.endswith(".npz") and not fn.endswith(".tmp.npz"):
            z = np.load(os.path.join(shard_dir, fn), allow_pickle=False)
            ids = z["ids"].tolist()
            for i, ex_id in enumerate(ids):
                out[ex_id] = {k: z[k][i] for k in z.files if k != "ids"}
    return out
