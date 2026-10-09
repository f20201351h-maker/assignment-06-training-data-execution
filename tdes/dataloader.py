"""Deterministic, resumable data planner.

Everything that decides which tokens the model sees next is derived from (config, compiled schedule, shard
registry, seed) or stored in `Planner.state`, a small JSON dict saved inside every checkpoint:

  streams["lane|pool|subpool"]  epoch (pass - 1), index into that epoch's document order, token offset inside
                                the current document, best-fit look-ahead buffer
  deferred[lane]                OPUS-deferred candidates (spans, row length, deferral count, first step)
  steps_done, token_positions_seen, loss_tokens_seen

Document order for an epoch is sorted by sha256(seed|lane|pool|subpool|epoch|shard|doc) (the role of a
Megatron-style shuffle index), so there is no hidden RNG state. The only other input is the model, which the
checkpoint restores bit for bit, because OPUS scores depend on it.
"""
import copy

import numpy as np
import torch

from . import opus
from .hashing import sha256_json, stable_u64
from .packing import batch_identity, materialize_row, row_hashes, sample_id, span_id

BEST_FIT = {"best_fit_whole_files", "structure_preserving_best_fit", "long_doc_best_fit"}


class LaneStream:
    def __init__(self, key, store, shard_ids, seed, state):
        self.key, self.seed = key, seed
        self.docs = [(sid, di["doc_id"], di["length"]) for sid in sorted(shard_ids)
                     for di in store.manifest(sid)["doc_index"]]
        if not self.docs:
            raise ValueError(f"empty stream {key}")
        self._orders = {}
        self.state = state

    def order(self, epoch):
        if epoch not in self._orders:
            self._orders[epoch] = sorted(self.docs, key=lambda d: (stable_u64(self.seed, self.key, epoch, d[0], d[1]),
                                                                   d[0], d[1]))
        return self._orders[epoch]

    def _cur(self):
        return self.order(self.state["epoch"])[self.state["idx"]]

    def _advance(self):
        s = self.state
        s["idx"], s["tok_off"] = s["idx"] + 1, 0
        if s["idx"] == len(self.docs):
            s["epoch"], s["idx"] = s["epoch"] + 1, 0

    def next_row(self, policy, L, window=8):
        s, spans = self.state, []
        if policy == "concat_chop":
            rem = L
            while rem > 0:
                sid, did, n = self._cur()
                take = min(rem, n - s["tok_off"])
                spans.append({"shard_id": sid, "doc_id": did, "start": s["tok_off"], "end": s["tok_off"] + take,
                              "pass": s["epoch"] + 1})
                s["tok_off"] += take
                rem -= take
                if s["tok_off"] == n:
                    self._advance()
        elif policy == "whole_trace_next_fit":
            rem = L
            while True:
                sid, did, n = self._cur()
                if n > L:
                    raise ValueError(f"{did} ({n}) longer than the row ({L})")
                if n > rem:
                    break
                spans.append({"shard_id": sid, "doc_id": did, "start": 0, "end": n, "pass": s["epoch"] + 1})
                rem -= n
                self._advance()
        elif policy in BEST_FIT:
            while len(s["buffer"]) < window:
                sid, did, n = self._cur()
                if n > L:
                    raise ValueError(f"{did} ({n}) longer than the row ({L})")
                s["buffer"].append({"shard_id": sid, "doc_id": did, "length": n, "pass": s["epoch"] + 1})
                self._advance()
            rem = L
            while True:
                fit = [b for b in s["buffer"] if b["length"] <= rem]
                if not fit:
                    break
                b = max(fit, key=lambda b: b["length"])  # max() keeps the first on ties
                s["buffer"].remove(b)
                spans.append({"shard_id": b["shard_id"], "doc_id": b["doc_id"], "start": 0, "end": b["length"],
                              "pass": b["pass"]})
                rem -= b["length"]
        else:
            raise ValueError(policy)
        return spans


def rows_to_batch(rows):
    keys = ["tokens", "labels", "loss_mask", "segment_ids", "position_ids"]
    return {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in keys}


def candidate_id(spans, L):
    return "cand-" + sha256_json([L, [[s["shard_id"], s["doc_id"], s["start"], s["end"], s["pass"]] for s in spans]])[:16]


class Planner:
    def __init__(self, cfg, plan, schedule, store, firewall, tokenizer, branch, state=None):
        self.cfg, self.plan, self.sched, self.store, self.fw, self.tok = cfg, plan, schedule, store, firewall, tokenizer
        self.branch = branch
        self.lanes = list(plan["lanes"])
        self.state = state or {"streams": {}, "deferred": {l: [] for l in self.lanes}, "steps_done": 0,
                               "token_positions_seen": 0, "loss_tokens_seen": 0}
        self._streams = {}
        self._build_proxy()

    def state_hash(self):
        return sha256_json(self.state)

    def snapshot(self):
        return copy.deepcopy(self.state)

    def stream(self, lane, pool, sub):
        key = f"{lane}|{pool}|{sub}"
        if key not in self._streams:
            st = self.state["streams"].setdefault(key, {"epoch": 0, "idx": 0, "tok_off": 0, "buffer": []})
            self._streams[key] = LaneStream(key, self.store, self.sched["lanes"][lane]["sources"][pool][sub],
                                            self.cfg["seed"], st)
        return self._streams[key]

    def _build_proxy(self):
        L = self.cfg["opus"]["proxy_row_len"]
        rows = []
        for sid in self.store.shards_where(split="proxy"):
            for di in self.store.manifest(sid)["doc_index"]:
                rows.append(materialize_row([{"shard_id": sid, "doc_id": di["doc_id"], "start": 0,
                                              "end": min(di["length"], L)}], self.store, L, self.tok.pad_id))
        self.proxy_rows = rows
        self.proxy_hash = sha256_json([row_hashes(r)["row_hash"] for r in rows])

    def _score(self, model, cands, step):
        """Scores grouped by row length (a batch needs one shape); proxy direction computed once per step."""
        with torch.no_grad():
            for sid in self.store.shards_where(split="proxy"):  # every proxy read is logged (no gradients)
                self.fw.eval_read(sid, step, self.branch, "opus_proxy_direction", permission="proxy_read")
            gp = opus.proxy_direction(model, rows_to_batch(self.proxy_rows))
            for L in sorted({c["row"]["L"] for c in cands}):
                grp = [c for c in cands if c["row"]["L"] == L]
                sc, gn = opus.opus_scores(model, rows_to_batch([c["row"] for c in grp]), gp)
                for c, s, g in zip(grp, sc, gn):
                    c["score"], c["head_grad_norm"] = s, g

    # ---- one step ---------------------------------------------------------------------------------------
    def plan_step(self, step, model, model_hash, drill_inject=False):
        st = self.sched["steps"][step - 1]
        assert st["step"] == step
        stage, quotas = st["stage"], st["quotas"]
        cands, pre = [], []
        for l in self.lanes:
            q = quotas[l]
            if q == 0:
                continue
            pool, L = st["pools"][l], st["row_len"][l]
            policy = self.sched["lanes"][l]["policy"]
            window = self.cfg["lanes"][l].get("best_fit_window", 8)
            size = opus.n_candidates(l, q, stage, self.plan)
            keep = []
            for e in self.state["deferred"][l]:
                if e["pool"] != pool or e["L"] != L:
                    pre.append(dict(e, status="rejected", reason="stage_mismatch",
                                    note=f"deferred for {e['pool']}/L{e['L']}, lane now draws {pool}/L{L}"))
                elif sum(1 for c in cands if c["lane"] == l) < size:
                    cands.append({"lane": l, "pool": pool, "subpool": e["subpool"], "L": L, "spans": e["spans"],
                                  "deferrals": e["deferrals"], "first_step": e["first_step"], "source": "deferred_queue"})
                else:
                    keep.append(e)
            self.state["deferred"][l] = keep
            n_fresh = size - sum(1 for c in cands if c["lane"] == l)
            subq = st["subquotas"][l]
            if opus.lane_mode(l, stage, self.plan) == "selector":
                if len(subq) != 1:
                    raise ValueError(f"selector lane {l} must have a single sub-pool, got {subq}")
                subs = [next(iter(subq))] * n_fresh
            else:
                subs = [b for b in sorted(subq) for _ in range(subq[b])][:n_fresh]
            for b in subs:
                spans = self.stream(l, pool, b).next_row(policy, L, window)
                cands.append({"lane": l, "pool": pool, "subpool": b, "L": L, "spans": spans, "deferrals": 0,
                              "first_step": step, "source": "stream"})
        if drill_inject:  # a misconfigured feeder offers test and validation rows as web candidates
            for split in ("test", "validation"):
                sid = self.store.shards_where(split=split)[0]
                di = self.store.manifest(sid)["doc_index"][0]
                L = st["row_len"]["web"]
                cands.append({"lane": "web", "pool": st["pools"]["web"], "subpool": "all", "L": L, "deferrals": 0,
                              "first_step": step, "source": f"DRILL:misconfigured {split} injection",
                              "spans": [{"shard_id": sid, "doc_id": di["doc_id"], "start": 0,
                                         "end": min(di["length"], L), "pass": 1}]})
        blocked, clean = [], []
        for c in cands:
            v = self.fw.row_violations(c)
            if v:
                blocked.append(self.fw.block("candidate_gate", step, c["source"], v, branch=self.branch,
                                             candidate_id=candidate_id(c["spans"], c["L"]),
                                             span_ids=[span_id(s) for s in c["spans"]],
                                             action="candidate_dropped_before_scoring"))
                pre.append(dict(c, status="rejected", reason="eval_firewall", note="; ".join(v[:3])))
            else:
                clean.append(c)
        cands = clean
        for i, c in enumerate(cands):
            c["idx"] = i
            c["row"] = materialize_row(c["spans"], self.store, c["L"], self.tok.pad_id)
        self._score(model, cands, step)
        dec, info = opus.decide(cands, quotas, stage, self.plan, self.cfg["opus"])
        # ---- records + deferral queue --------------------------------------------------------------------
        records, n = [], 0
        for e in pre:
            n += 1
            records.append(self._record(step, n, st, model_hash, e, None, None, e["status"], e["reason"], info))
        accepted = []
        qmax = self.cfg["opus"]["max_deferred_per_lane"]
        lane_pos = {l: i for i, l in enumerate(self.lanes)}
        for c in sorted(cands, key=lambda c: (lane_pos[c["lane"]], info["rank_in_lane"][c["idx"]])):
            status, reason = dec[c["idx"]]
            if status == "deferred":
                qd = self.state["deferred"][c["lane"]]
                if len(qd) >= qmax:
                    status, reason = "rejected", "defer_queue_full"
                else:
                    qd.append({"lane": c["lane"], "pool": c["pool"], "subpool": c["subpool"], "L": c["L"],
                               "spans": c["spans"], "deferrals": c["deferrals"] + 1, "first_step": c["first_step"]})
            n += 1
            rec = self._record(step, n, st, model_hash, c, c["score"], info["rank_in_lane"][c["idx"]], status,
                               reason, info)
            records.append(rec)
            if status == "accepted":
                accepted.append((c, rec))
        got = {l: sum(1 for c, _ in accepted if c["lane"] == l) for l in self.lanes}
        if got != quotas:
            raise RuntimeError(f"step {step}: accepted rows {got} != compiled quotas {quotas}")
        rows = []
        for c, rec in accepted:
            rows.append({"lane": c["lane"], "pool": c["pool"], "subpool": c["subpool"],
                         "policy": self.sched["lanes"][c["lane"]]["policy"], "spans": c["spans"],
                         "candidate_id": rec["candidate_id"], "decision_id": rec["decision_id"],
                         "opus_score": c["score"], "head_grad_norm": c["head_grad_norm"],
                         "accept_reason": rec["reason"], **c["row"]})
        # regular rows first (lane order), long rows last; microbatches of `microbatch_positions` positions
        rows.sort(key=lambda r: (r["L"] == self.plan["long_row_len"], lane_pos[r["lane"]]))
        mbp = self.plan["microbatch_positions"]
        mbs, cur = [], []
        for r in rows:
            if cur and (cur[0]["L"] != r["L"] or sum(x["L"] for x in cur) + r["L"] > mbp):
                mbs.append(cur)
                cur = []
            cur.append(r)
        if cur:
            mbs.append(cur)
        self.state["steps_done"] = step
        self.state["token_positions_seen"] += sum(r["L"] for r in rows)
        self.state["loss_tokens_seen"] += sum(r["n_loss"] for r in rows)
        return {"rows": rows, "microbatches": mbs, "decisions": records, "blocked": blocked, "stage": stage,
                "quotas": quotas, "cutoff_score": info["cutoff_score"]}

    def _record(self, step, n, st, model_hash, c, score, rank, status, reason, info):
        row = c.get("row")
        return {
            "decision_id": f"opus-{self.branch}-s{step:05d}-{n:03d}", "candidate_id": candidate_id(c["spans"], c["L"]),
            "step": step, "branch": self.branch, "lane": c["lane"], "stage": st["stage"], "pool": c["pool"],
            "subpool": c.get("subpool"), "row_len": c["L"], "source": c.get("source", "deferred_queue"),
            "lane_mode": opus.lane_mode(c["lane"], st["stage"], self.plan),
            "shard_ids": sorted({s["shard_id"] for s in c["spans"]}), "span_ids": [span_id(s) for s in c["spans"]],
            "passes": sorted({s["pass"] for s in c["spans"]}),
            "scoring_model": {"after_step": step - 1, "model_sha256": model_hash},
            "proxy_version": {"name": self.cfg["opus"]["proxy_version"], "proxy_batch_sha256": self.proxy_hash},
            "opus_score": score, "rank_in_lane": rank, "lane_quota": st["quotas"][c["lane"]],
            "web_cutoff_score": info["cutoff_score"], "status": status, "reason": reason,
            "protected_floor_override": reason == "protected_floor_override",
            "lane_protected": c["lane"] in self.plan["protected"], "deferrals_before": c["deferrals"],
            "first_seen_step": c["first_step"], "note": c.get("note"),
            "n_loss_tokens": row["n_loss"] if row is not None else None,
            "n_real_tokens": row["n_real"] if row is not None else None,
            "effective_token_estimate": (round(row["n_loss"] * max(score, 0.0), 3) if score is not None else None),
            "head_grad_norm": c.get("head_grad_norm"),
        }

    def identify(self, step, rows):
        for r in rows:
            r.update(row_hashes(r))
            r["sample_id"] = sample_id(r["spans"], r["L"])
        return batch_identity(step, rows)


def preview_next_batch(cfg, plan, sched, store, fw, tok, branch, state, model, model_hash, step):
    """Plan `step` on a copy of the dataloader state with a silent firewall; nothing on disk or in `state` changes.
    The result is what the checkpoint declares as its expected next batch."""
    was = fw.silent
    fw.silent = True
    try:
        p = Planner(cfg, plan, sched, store, fw, tok, branch, state=copy.deepcopy(state))
        out = p.plan_step(step, model, model_hash, drill_inject=False)
        ident = p.identify(step, out["rows"])
        return {"step": step, **ident, "sample_ids": [r["sample_id"] for r in out["rows"]],
                "span_ids": [[span_id(s) for s in r["spans"]] for r in out["rows"]],
                "row_hashes": [r["row_hash"] for r in out["rows"]]}
    finally:
        fw.silent = was
