"""Packing / masks / positions, the A5 quota compiler and protected floors, OPUS decisions, firewall gates."""
import json
import math
import random

import numpy as np
import pytest
import torch

from tdes.context import REPO, load_config
from tdes.contracts import load_a5_plan
from tdes.dataloader import LaneStream
from tdes.model import TinyGPT, build_attn_mask
from tdes.opus import decide
from tdes.packing import IGNORE, attention_mask, materialize_row
from tdes.schedule import compile_quotas

PAD, EOS, L = 10000, 10001, 16


class FakeStore:
    def __init__(self, docs):
        self.docs = docs

    def doc_tokens(self, shard_id, doc_id):
        t, e = self.docs[doc_id]
        return np.asarray(t), np.asarray(e)

    def manifest(self, sid):
        return {"doc_index": [{"doc_id": d, "length": len(t)} for d, (t, e) in sorted(self.docs.items())]}


def sp(doc, s, e):
    return {"shard_id": "x", "doc_id": doc, "start": s, "end": e, "pass": 1}


def naive(spans, store, L):
    toks, elig, seg = [], [], []
    for k, s in enumerate(spans, 1):
        t, e = store.doc_tokens(None, s["doc_id"])
        toks += list(t[s["start"]:s["end"]])
        elig += list(e[s["start"]:s["end"]])
        seg += [k] * (s["end"] - s["start"])
    n = len(toks)
    toks, elig, seg = toks + [PAD] * (L - n), elig + [0] * (L - n), seg + [0] * (L - n)
    pos, last, c = [], None, 0
    for s in seg:
        c = c + 1 if (s == last and s != 0) else 0
        pos.append(c if s else 0)
        last = s
    lab = [toks[i + 1] if i + 1 < L and seg[i] and seg[i] == seg[i + 1] and elig[i + 1] else IGNORE for i in range(L)]
    return toks, lab, seg, pos


def test_packer_matches_naive_reference_on_random_rows():
    rng = random.Random(0)
    for _ in range(300):
        docs = {}
        for i in range(10):
            n = rng.randint(1, 20)
            docs[f"d{i}"] = ([rng.randint(0, 9999) for _ in range(n - 1)] + [EOS], [rng.randint(0, 1) for _ in range(n)])
        store = FakeStore(docs)
        spans, room = [], L
        for d, (t, _) in docs.items():
            if room == 0:
                break
            s = rng.randint(0, len(t) - 1)
            e = min(len(t), s + room)
            spans.append(sp(d, s, e))
            room -= e - s
        r = materialize_row(spans, store, L, PAD)
        toks, lab, seg, pos = naive(spans, store, L)
        assert list(r["tokens"]) == toks and list(r["labels"]) == lab
        assert list(r["segment_ids"]) == seg and list(r["position_ids"]) == pos
        assert r["n_context_only"] == r["n_real"] - r["n_loss"]


def test_mask_edge_cases():
    store = FakeStore({"full": (list(range(15)) + [EOS], [1] * 16), "one": ([EOS], [1]),
                       "ctx": ([1, 2, 3, EOS], [0, 0, 0, 0]), "a": ([5, 6, EOS], [1, 1, 1])})
    r = materialize_row([sp("full", 0, 16)], store, L, PAD)
    assert r["n_real"] == L and r["n_loss"] == L - 1 and r["labels"][-1] == IGNORE
    r = materialize_row([sp("one", 0, 1), sp("a", 0, 3)], store, L, PAD)
    assert r["labels"][0] == IGNORE and list(r["position_ids"][:4]) == [0, 0, 1, 2]  # no loss across the boundary
    assert materialize_row([sp("ctx", 0, 4)], store, L, PAD)["n_loss"] == 0  # context-only sample
    r = materialize_row([sp("a", 0, 3)], store, L, PAD)
    assert (r["loss_mask"][3:] == 0).all() and (r["segment_ids"][3:] == 0).all() and (r["tokens"][3:] == PAD).all()
    with pytest.raises(ValueError):
        materialize_row([sp("full", 0, 16), sp("a", 0, 1)], store, L, PAD)


def test_attention_is_block_diagonal_causal_and_matches_the_model():
    seg = np.array([1, 1, 1, 2, 2, 0, 0, 0])
    m = attention_mask(seg)
    for i in range(8):
        for j in range(8):
            assert m[i, j] == ((j <= i and seg[i] == seg[j] and seg[i] > 0) or i == j)
    assert (build_attn_mask(torch.tensor(seg)[None])[0].numpy() == m).all()


def test_model_cannot_see_across_segments():
    torch.manual_seed(0)
    model = TinyGPT(300, 16, 32, 2, 4, 64).eval()
    toks = torch.randint(0, 255, (1, 16))
    seg = torch.tensor([[1] * 6 + [2] * 6 + [0] * 4])
    pos = torch.tensor([[*range(6), *range(6), 0, 0, 0, 0]])
    out1 = model(toks, pos, seg)
    t2 = toks.clone()
    t2[0, :6] = torch.randint(0, 255, (6,))
    out2 = model(t2, pos, seg)
    assert torch.allclose(out1[0, 6:12], out2[0, 6:12]) and not torch.allclose(out1[0, :6], out2[0, :6])


def test_policies_whole_documents_and_concat():
    docs = {f"d{i}": (list(range(n - 1)) + [EOS], [1] * n) for i, n in enumerate([5, 9, 3, 12, 7, 4])}
    store = FakeStore(docs)
    st = {"epoch": 0, "idx": 0, "tok_off": 0, "buffer": []}
    ls = LaneStream("t", store, ["x"], 1, st)
    for _ in range(4):
        row = ls.next_row("best_fit_whole_files", L, 4)
        assert sum(s["end"] - s["start"] for s in row) <= L
        assert all(s["start"] == 0 and s["end"] == len(docs[s["doc_id"]][0]) for s in row)  # never cut
    st2 = {"epoch": 0, "idx": 0, "tok_off": 0, "buffer": []}
    ls2 = LaneStream("t", store, ["x"], 1, st2)
    rows = [ls2.next_row("concat_chop", L) for _ in range(5)]
    assert all(sum(s["end"] - s["start"] for s in r) == L for r in rows)  # always full
    assert st2["epoch"] >= 1  # wrapped into a second pass, recorded in the spans
    assert any(s["pass"] == 2 for r in rows for s in r)


@pytest.fixture(scope="module")
def plan_q():
    cfg = load_config()
    plan = load_a5_plan(cfg, REPO)
    return cfg, plan, compile_quotas(plan, cfg)


def test_quotas_fill_every_microbatch_and_follow_stages(plan_q):
    cfg, plan, q = plan_q
    for st in q:
        reg = [l for l in plan["lanes"] if cfg["lanes"][l]["row"] == "base"]
        assert sum(st["quotas"][l] for l in reg) == st["regular_rows"] == 2048 // st["base_window"]
    s1 = [st for st in q if st["stage"] == "S1_foundation" and st["ramp"] is None]
    assert all(st["quotas"]["agentic"] == st["quotas"]["reasoning"] == st["quotas"]["long_context"] == 0 for st in s1)


def test_protected_floor_holds_at_every_step_on_actual_positions(plan_q):
    cfg, plan, q = plan_q
    for st in q:
        tot = st["segment_cum_total_positions"]
        for l in plan["protected"]:
            assert st["segment_cum_quota_positions"][l] >= st["stage_floor_pct"][l] / 100 * tot - 1e-9


def test_ramps_and_window_move_first(plan_q):
    cfg, plan, q = plan_q
    s3 = next(s for s in plan["stages"] if s["name"] == "S3_reasoning_code")["step_start"]
    ramp = [st["step"] for st in q if st["ramp"]]
    assert s3 - 1 in ramp and s3 in ramp
    first256 = min(st["step"] for st in q if st["base_window"] == 256)
    assert first256 < s3 - 1  # the window moves before the mixture starts to ramp


def test_fork_override_changes_only_later_steps(plan_q):
    cfg, plan, q = plan_q
    ov = {"from_step": 15, "mix": cfg["fork"]["mix_overrides"]}
    qf = compile_quotas(plan, cfg, ov)
    assert [s["quotas"] for s in q[:14]] == [s["quotas"] for s in qf[:14]]
    assert sum(s["quotas"]["indic"] for s in qf[14:20]) > sum(s["quotas"]["indic"] for s in q[14:20])


def _plan():
    return {"protected": ["indic", "reasoning", "agentic"], "selector_keep": {"web": 0.4, "code": 1.0, "stem": 1.0}}


OPUS_CFG = {"defer_band_factor": 0.5, "max_deferrals": 2}


def test_opus_selector_defers_and_rejects():
    cands = [{"idx": i, "lane": "web", "score": s, "deferrals": 0} for i, s in enumerate([.9, .8, .7, .6, .5])]
    out, info = decide(cands, {"web": 2}, "S1_foundation", _plan(), OPUS_CFG)
    assert [out[i][0] for i in range(5)] == ["accepted", "accepted", "deferred", "rejected", "rejected"]
    assert info["cutoff_score"] == .8
    cands[2]["deferrals"] = 2
    out, _ = decide(cands, {"web": 2}, "S1_foundation", _plan(), OPUS_CFG)
    assert out[2] == ("rejected", "defer_expired")


def test_opus_protected_override_and_bypass_and_anneal():
    cands = [{"idx": 0, "lane": "web", "score": .5, "deferrals": 0}, {"idx": 1, "lane": "indic", "score": .1, "deferrals": 0},
             {"idx": 2, "lane": "indic", "score": .9, "deferrals": 0}, {"idx": 3, "lane": "code", "score": -1, "deferrals": 0},
             {"idx": 4, "lane": "long_context", "score": -1, "deferrals": 0}]
    q = {"web": 1, "indic": 2, "code": 1, "long_context": 1}
    out, _ = decide(cands, q, "S4_long_context", _plan(), OPUS_CFG)
    assert out[1] == ("accepted", "protected_floor_override") and out[2] == ("accepted", "always_on_above_cutoff")
    assert out[3] == ("accepted", "keep_1.0_static_filter_only") and out[4] == ("accepted", "selector_bypass_long_context")
    out, _ = decide(cands, q, "anneal", _plan(), OPUS_CFG)
    assert all(v == ("accepted", "selector_off_anneal") for v in out.values())


def test_firewall_gates(built, tok):
    from tdes.firewall import Firewall, FirewallViolation
    from tdes.shards import ShardStore
    st = ShardStore(built, tok.tokenizer_id)
    fw = Firewall(built, st, tok)
    test_sid, val_sid = st.shards_where(split="test")[0], st.shards_where(split="validation")[0]
    assert fw.shard_violations(test_sid) and fw.shard_violations(val_sid)
    assert not fw.shard_violations(st.shards_where(split="train")[0])
    di = st.manifest(test_sid)["doc_index"][0]
    text = tok.decode(st.doc_tokens(test_sid, di["doc_id"])[0])
    assert fw.text_violations("some web text. " + text)  # eval text inside a "train" document is still caught
    row = {"spans": [{"shard_id": val_sid, "doc_id": st.manifest(val_sid)["doc_index"][0]["doc_id"], "start": 0, "end": 8}]}
    with pytest.raises(FirewallViolation):
        fw.check_batch([row], 1, "t")
    with pytest.raises(FirewallViolation):  # validation may only be read with gradients disabled
        fw.eval_read(val_sid, 1, "t", "test")
    with torch.no_grad():
        fw.eval_read(val_sid, 1, "t", "test")
    with pytest.raises(FirewallViolation):
        with torch.no_grad():
            fw.eval_read(test_sid, 1, "t", "test")
