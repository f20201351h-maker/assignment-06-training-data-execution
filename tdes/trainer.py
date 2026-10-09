"""Training loop: plan -> firewall -> train -> learning ledger -> consumption ledger -> checkpoint.

Run as a child process by run_demo.py so that the deliberate crash is a real process death (os._exit: no
cleanup, no finally blocks, nothing survives in memory):

  python -m tdes.trainer --artifacts A --run-root R --branch main --mode fresh   [--crash]
  python -m tdes.trainer ... --mode resume
  python -m tdes.trainer ... --mode fork

Per step, files are written in this order: OPUS decisions, learning ledger (token CSV + sample JSONL),
performance row, then the consumption-ledger record (the commit point), then a checkpoint every
`checkpoint_every` steps. A checkpoint records the ledger offset/head hash and the batch that must come next.
The model is deliberately tiny (2 layers, d=64, A2 vocabulary); it exists to produce real per-token losses.
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import runlog
from .checkpoint import ckpt_id, list_checkpoints, load_checkpoint, save_checkpoint
from .context import Ctx, load_schedule, setup_determinism
from .dataloader import Planner, preview_next_batch, rows_to_batch
from .firewall import Firewall, FirewallViolation
from .hashing import sha256_bytes, sha256_json
from .ledger import HashChainLedger, read_ledger, recover_to_checkpoint, truncate_jsonl_by_step
from .model import TinyGPT, optimizer_hash, state_hash, token_losses
from .packing import materialize_row, span_id
from .schedule import compile_schedule

TOKEN_COLUMNS = ["step", "microbatch", "row", "target_pos", "sample_id", "span_index", "shard_id", "doc_id",
                 "doc_offset", "token_id", "token_preview", "lane", "language", "is_special", "is_eos", "loss", "ppl",
                 "loss_after_update", "repeat_pass"]
# per-token facts only; checkpoint before/after, model age, stage, phase and the OPUS decision of every token are
# on its sample line in learning_samples.jsonl (join on sample_id + span_index)
ATTENTION_POLICY = "causal, block-diagonal by segment_id (no cross-document attention, padding isolated)"
POSITION_POLICY = "position_ids restart at 0 at every segment start"


def lr_factor(cfg, plan):
    """Warm-up, constant through the main run, linear decay towards zero in the anneal (A5 section 7)."""
    W = cfg["model"]["warmup_steps"]
    an = plan["stages"][-1]

    def f(i):  # i = optimizer steps already taken; this update is step i + 1
        s = i + 1
        if s <= W:
            return s / W
        if s >= an["step_start"]:
            return (an["step_end"] - s + 1) / (an["n_steps"] + 1)
        return 1.0
    return f


def build_model(cfg, plan, vocab):
    m = cfg["model"]
    torch.manual_seed(m["init_seed"])
    model = TinyGPT(vocab, plan["long_row_len"], **m)
    opt = torch.optim.AdamW(model.parameters(), lr=m["lr"], betas=tuple(m["betas"]), weight_decay=m["weight_decay"])
    return model, opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_factor(cfg, plan))


def phase_of(plan, stage, step):
    if stage == "anneal":
        return "anneal"
    f = step / plan["stages"][-2]["step_end"]
    return "early" if f <= 1 / 3 else ("mid" if f <= 2 / 3 else "late")


def train_on_microbatches(model, opt, sched, mbs, cfg):
    """One optimizer step, gradients accumulated over microbatches. Returns per-row token losses."""
    n_loss = sum(int(r["n_loss"]) for mb in mbs for r in mb)
    opt.zero_grad(set_to_none=True)
    before = []
    for mb in mbs:
        tl = token_losses(model, rows_to_batch(mb))
        (tl.sum() / n_loss).backward()
        before.extend(tl.detach().numpy().astype(np.float32))
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["model"]["grad_clip"]))
    lr = float(sched.get_last_lr()[0])
    opt.step()
    sched.step()
    after = []
    with torch.no_grad():
        for mb in mbs:
            after.extend(token_losses(model, rows_to_batch(mb)).numpy().astype(np.float32))
    loss = float(np.float64(sum(float(x.astype(np.float64).sum()) for x in before)) / n_loss)
    return {"before": before, "after": after, "grad_norm": gn, "lr": lr, "n_loss": n_loss, "step_loss": loss}


def validation_loss(model, ctx, fw, step, branch):
    """Held-out loss per language on the A2 held-out validation shard; gradients off, every read logged."""
    store, L = ctx.store, ctx.cfg["opus"]["proxy_row_len"]
    out = {}
    with torch.no_grad():
        for sid in store.shards_where(split="validation"):
            fw.eval_read(sid, step, branch, "checkpoint_validation_loss")
            by_lang = {}
            for di in store.manifest(sid)["doc_index"]:
                by_lang.setdefault(di["language"], []).append(materialize_row(
                    [{"shard_id": sid, "doc_id": di["doc_id"], "start": 0, "end": min(di["length"], L)}],
                    store, L, ctx.tok.pad_id))
            for lang, rows in sorted(by_lang.items()):
                tl = token_losses(model, rows_to_batch(rows))
                n = sum(r["n_loss"] for r in rows)
                out[lang] = {"loss": float(tl.sum()) / n, "n_loss_tokens": n}
    return out


class BranchRun:
    def __init__(self, ctx, run_root, branch, run_id, schedule, attempt, last_ckpt, drills=True):
        self.ctx, self.cfg, self.plan, self.tok, self.store = ctx, ctx.cfg, ctx.plan, ctx.tok, ctx.store
        self.run_root = Path(run_root)
        self.branch, self.run_id, self.sched, self.attempt = branch, run_id, schedule, attempt
        self.ldir = self.run_root / "ledgers" / branch
        self.ldir.mkdir(parents=True, exist_ok=True)
        self.ckroot = self.run_root / "checkpoints"
        self.fw = Firewall(ctx.artifacts, ctx.store, ctx.tok, self.ldir / "firewall_events.jsonl",
                           self.ldir / "eval_access_log.jsonl")
        self.last_ckpt, self.drills = last_ckpt, drills
        self.model, self.opt, self.lrs = build_model(self.cfg, self.plan, self.tok.vocab_size)
        self.planner = self.ledger = None

    def _append_jsonl(self, name, recs):
        with open(self.ldir / name, "a", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")

    def _learning(self, step, stage, mbs, res, ckb, cka, tokens_seen, batch_id):
        """Token-level trace (CSV) and sample-level summaries; returns the hash of the token lines."""
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        samples, phase, ri = [], phase_of(self.plan, stage, step), 0
        for k, mb in enumerate(mbs):
            mbid = f"{batch_id}/mb{k}"
            for r in mb:
                before, after = res["before"][ri], res["after"][ri]
                starts, s0 = [], 0
                for sp in r["spans"]:
                    starts.append(s0)
                    s0 += sp["end"] - sp["start"]
                per_span = {}
                for i in np.nonzero(r["loss_mask"])[0]:
                    tp = int(i) + 1  # the target position: loss at i scores the prediction of token tp
                    j = int(r["segment_ids"][tp]) - 1
                    sp = r["spans"][j]
                    di = self.store.doc_info(sp["shard_id"], sp["doc_id"])
                    tid = int(r["tokens"][tp])
                    lo, la = float(before[i]), float(after[i])
                    w.writerow([step, k, ri, tp, r["sample_id"], j, sp["shard_id"], sp["doc_id"],
                                sp["start"] + (tp - starts[j]), tid,
                                json.dumps(self.tok.token_preview(tid), ensure_ascii=False)[1:-1],
                                r["lane"], di["language"], int(self.tok.is_special(tid)), int(tid == self.tok.eos_id),
                                repr(lo), f"{math.exp(lo):.6g}", repr(la), sp["pass"]])
                    per_span.setdefault(j, []).append((tp, tid, lo, la))
                for j, sp in enumerate(r["spans"]):
                    toks = per_span.get(j, [])
                    n = len(toks)
                    sb, sa = sum(x[2] for x in toks), sum(x[3] for x in toks)
                    delta = (sa - sb) / n if n else 0.0
                    samples.append({
                        "branch": self.branch, "step": step, "attempt": self.attempt, "row": ri, "microbatch_id": mbid,
                        "sample_id": r["sample_id"], "span_index": j, "span_id": span_id(sp), "shard_id": sp["shard_id"],
                        "doc_id": sp["doc_id"], "lane": r["lane"], "subpool": r["subpool"], "stage": stage,
                        "phase": phase, "repeat_pass": sp["pass"], "n_loss_tokens": n, "loss_sum": sb,
                        "mean_loss": sb / n if n else None,
                        "mean_token_ppl": (sum(math.exp(x[2]) for x in toks) / n) if n else None,
                        "loss_after_sum": sa, "loss_delta": delta,
                        "high_perplexity_tokens": [{"target_pos": x[0], "token": self.tok.token_preview(x[1]),
                                                    "loss": x[2]} for x in sorted(toks, key=lambda x: -x[2])[:3]],
                        "opus_decision_id": r["decision_id"], "opus_score": r["opus_score"],
                        "accept_reason": r["accept_reason"], "row_head_grad_norm": r["head_grad_norm"],
                        "step_grad_norm": res["grad_norm"], "model_tokens_seen_before": tokens_seen,
                        "checkpoint_before": ckb, "checkpoint_after": cka,
                        "classification": ("no_signal" if not n else "useful" if delta < -0.05
                                           else "harmful" if delta > 0.05 else "neutral")})
                ri += 1
        text = buf.getvalue()
        path = self.ldir / "learning_tokens.csv"
        new = not path.exists()
        with open(path, "a", encoding="utf-8", newline="") as f:
            if new:
                f.write(",".join(TOKEN_COLUMNS) + "\n")
            f.write(text)
        self._append_jsonl("learning_samples.jsonl", samples)
        return sha256_bytes(text.encode("utf-8"))

    def step(self, step, crash=None):
        cfg = self.cfg
        t0 = time.perf_counter()
        mh = state_hash(self.model.state_dict())
        dl_before = self.planner.state_hash()
        tokens_seen = self.planner.state["token_positions_seen"]
        drills = cfg["firewall_drills"] if self.drills else {}
        plan = self.planner.plan_step(step, self.model, mh, drill_inject=(step == drills.get("candidate_injection_step")))
        rows, mbs = plan["rows"], plan["microbatches"]
        ident = self.planner.identify(step, rows)
        for b in plan["blocked"]:
            runlog.log(self.ctx.artifacts, f"[{self.branch}] evaluation data blocked at the candidate gate, step {step}: "
                                           f"{b['attempt']} -> {', '.join(b['reasons'][:2])}")
        t1 = time.perf_counter()
        drill = None
        if step == drills.get("direct_batch_injection_step"):
            vs = self.store.shards_where(split="validation")[0]
            di = self.store.manifest(vs)["doc_index"][0]
            L = rows[0]["L"]
            sp = [{"shard_id": vs, "doc_id": di["doc_id"], "start": 0, "end": min(di["length"], L), "pass": 1}]
            bad = dict(materialize_row(sp, self.store, L, self.tok.pad_id), spans=sp, lane="web")
            try:
                self.fw.check_batch(rows[:-1] + [bad], step, self.branch)
                drill = "NOT BLOCKED"
            except FirewallViolation:
                drill = "blocked"
            runlog.log(self.ctx.artifacts, f"[{self.branch}] firewall drill step {step}: a validation row placed directly "
                                           f"into the loss-bearing batch -> {drill}")
        self.fw.check_batch(rows, step, self.branch)  # the real batch must pass
        t2 = time.perf_counter()
        res = train_on_microbatches(self.model, self.opt, self.lrs, mbs, cfg)
        t3 = time.perf_counter()
        E = cfg["checkpoint_every"]
        ckb, cka = self.last_ckpt, ckpt_id(self.branch, math.ceil(step / E) * E)
        self._append_jsonl("opus_decisions.jsonl", plan["decisions"])
        trace_hash = self._learning(step, plan["stage"], mbs, res, ckb, cka, tokens_seen, ident["batch_id"])
        st = self.sched["steps"][step - 1]
        counts = {}
        for d in plan["decisions"]:
            k = d["status"] + ("/protected_floor_override" if d["protected_floor_override"] else "")
            counts[k] = counts.get(k, 0) + 1
        ri = 0
        mbrecs = []
        for k, mb in enumerate(mbs):
            rr = []
            for r in mb:
                rr.append({"row": ri, "sample_id": r["sample_id"], "lane": r["lane"], "subpool": r["subpool"],
                           "policy": r["policy"], "pool": r["pool"], "L": r["L"], "candidate_id": r["candidate_id"],
                           "opus_decision_id": r["decision_id"], "accept_reason": r["accept_reason"],
                           "opus_score": r["opus_score"], "shard_ids": sorted({s["shard_id"] for s in r["spans"]}),
                           "spans": r["spans"], "row_hash": r["row_hash"], "tokens_hash": r["tokens_hash"],
                           "loss_mask_hash": r["loss_mask_hash"], "position_ids_hash": r["position_ids_hash"],
                           "attention_mask_hash": r["attention_mask_hash"], "n_real_tokens": r["n_real"],
                           "n_loss_tokens": r["n_loss"]})
                ri += 1
            mbrecs.append({"microbatch_id": f"{ident['batch_id']}/mb{k}", "rank": 0, "L": mb[0]["L"], "rows": rr})
        rec = {
            "run_id": self.run_id, "branch_id": self.branch, "attempt": self.attempt, "global_step": step,
            "stage": plan["stage"], "segment": st["segment"], "ramp": st["ramp"], "base_window": st["base_window"],
            "checkpoint_id_base": ckb, "next_checkpoint_id": cka, "rank": 0, "world_size": 1,
            "batch_id": ident["batch_id"], "batch_hash": ident["batch_hash"],
            "batch_loss_mask_hash": ident["batch_loss_mask_hash"], "tokenizer_id": self.tok.tokenizer_id,
            "a2_tokenizer_sha256": self.tok.a2_sha256, "dataloader_version": cfg["dataloader_version"],
            "schedule_sha256": self.sched["schedule_sha256"], "attention_policy": ATTENTION_POLICY,
            "position_policy": POSITION_POLICY, "microbatches": mbrecs,
            "lane_rows": {l: n for l, n in plan["quotas"].items() if n},
            "opus": {"decisions_sha256": sha256_json(plan["decisions"]), "n_decisions": len(plan["decisions"]),
                     "counts": counts, "web_cutoff_score": plan["cutoff_score"],
                     "firewall_blocked_candidates": len(plan["blocked"])},
            "step_loss": res["step_loss"], "n_loss_tokens": res["n_loss"],
            "n_real_tokens": int(sum(r["n_real"] for r in rows)), "token_positions": int(sum(r["L"] for r in rows)),
            "grad_norm": res["grad_norm"], "lr": res["lr"], "model_sha256_before": mh,
            "model_sha256_after": state_hash(self.model.state_dict()), "dataloader_state_sha256_before": dl_before,
            "dataloader_state_sha256_after": self.planner.state_hash(), "learning_trace_sha256": trace_hash,
            "model_tokens_seen_after": self.planner.state["token_positions_seen"], "firewall_drill": drill}
        t4 = time.perf_counter()
        perf = {"branch": self.branch, "step": step, "attempt": self.attempt, "t_plan_and_score_s": t1 - t0,
                "t_firewall_s": t2 - t1, "t_train_s": t3 - t2, "token_positions": rec["token_positions"],
                "real_tokens": rec["n_real_tokens"], "loss_tokens": res["n_loss"],
                "candidates_scored": sum(1 for d in plan["decisions"] if d["opus_score"] is not None),
                "candidate_positions_scored": sum(d["row_len"] for d in plan["decisions"] if d["opus_score"] is not None),
                "accepted_rows": len(rows)}
        if crash is not None and crash["step"] == step:
            committed = read_ledger(self.ledger.path)
            _, line = self.ledger.encode(rec)
            runlog.log(self.ctx.artifacts, f"[{self.branch}] crash injection: step {step} trained in memory; writing "
                                           f"{len(line) // 2} of {len(line)} bytes of its ledger record, then "
                                           f"os._exit({crash['exit_code']}) (last durable checkpoint {self.last_ckpt}, "
                                           f"committed ledger offset {committed['offset']})")
            with open(self.ledger.path, "ab") as f:  # torn write, then a hard exit: no cleanup runs
                f.write(line[:len(line) // 2])
                f.flush()
                os.fsync(f.fileno())
            os._exit(crash["exit_code"])
        self.ledger.append(rec)
        perf["t_ledger_s"] = time.perf_counter() - t4
        perf["t_step_total_s"] = time.perf_counter() - t0
        perf["t_checkpoint_s"] = 0.0
        if step % E == 0:
            t5 = time.perf_counter()
            self.save(step)
            perf["t_checkpoint_s"] = time.perf_counter() - t5
        self._append_jsonl("performance_steps.jsonl", [perf])
        return rec

    def save(self, step):
        vl = validation_loss(self.model, self.ctx, self.fw, step, self.branch)
        self._append_jsonl("eval_checkpoints.jsonl", [{"branch": self.branch, "step": step, "validation_loss": vl,
                                                       "grad_enabled": False}])
        nxt = None
        if step < len(self.sched["steps"]):
            nxt = preview_next_batch(self.cfg, self.plan, self.sched, self.store, self.fw, self.tok, self.branch,
                                     self.planner.state, self.model, state_hash(self.model.state_dict()), step + 1)
        meta = save_checkpoint(self.ckroot, self.branch, step, self.model, self.opt, self.lrs, self.planner.snapshot(),
                               self.ledger, {"run_id": self.run_id, "attempt": self.attempt,
                                             "tokens_seen": self.planner.state["token_positions_seen"],
                                             "tokenizer_id": self.tok.tokenizer_id,
                                             "schedule_sha256": self.sched["schedule_sha256"],
                                             "shard_registry_sha256": self.store.registry_sha256,
                                             "parent_checkpoint_id": self.last_ckpt, "validation_loss": vl,
                                             "next_batch": nxt})
        m2, o2, s2 = build_model(self.cfg, self.plan, self.tok.vocab_size)
        meta2, dstate = load_checkpoint(self.ckroot / self.branch / f"step_{step:05d}", m2, o2, s2)
        ok = (state_hash(m2.state_dict()) == state_hash(self.model.state_dict()) and optimizer_hash(o2) ==
              optimizer_hash(self.opt) and sha256_json(dstate) == self.planner.state_hash()
              and meta2["ledger_head_hash"] == self.ledger.head and meta2["ledger_offset"] == self.ledger.offset)
        runlog.log(self.ctx.artifacts, f"[{self.branch}] checkpoint saved: {meta['checkpoint_id']} ledger_offset="
                                       f"{meta['ledger_offset']} head={meta['ledger_head_hash'][:16]} model="
                                       f"{meta['model_sha256'][:16]} next_batch={nxt['batch_id'] if nxt else None} "
                                       f"val_loss={ {k: round(v['loss'], 3) for k, v in vl.items()} }")
        runlog.check(self.ctx.artifacts, "checkpoint_saved", ok, f"{meta['checkpoint_id']} reloaded into fresh objects; "
                     "model/optimizer/dataloader hashes and ledger binding re-verified")
        self.last_ckpt = meta["checkpoint_id"]

    def log_first_step(self, rec, step):
        by = {}
        for mb in rec["microbatches"]:
            for r in mb["rows"]:
                p = by.setdefault(f"{r['policy']}@{r['L']}", [0, 0, 0])
                p[0] += 1
                p[1] += r["n_real_tokens"]
                p[2] += r["n_loss_tokens"]
        runlog.log(self.ctx.artifacts, f"[{self.branch}] batches packed: step {step} {rec['batch_id']} "
                   f"{sum(len(mb['rows']) for mb in rec['microbatches'])} rows in {len(rec['microbatches'])} "
                   f"microbatches; " + ", ".join(f"{k}: {v[0]} rows util {v[1] / (v[0] * int(k.split('@')[1])):.3f} "
                                                 f"loss-tok {v[2]}" for k, v in sorted(by.items())))
        runlog.log(self.ctx.artifacts, f"[{self.branch}] OPUS decisions recorded: step {step} "
                                       f"{rec['opus']['n_decisions']} candidates {rec['opus']['counts']}; web cutoff "
                                       f"{rec['opus']['web_cutoff_score']:.4f}")

    def run(self, first, last, crash=None):
        for s in range(first, last + 1):
            rec = self.step(s, crash)
            if s == first:
                self.log_first_step(rec, s)
            if s == first or s % 7 == 0 or s == last:
                runlog.log(self.ctx.artifacts, f"[{self.branch}] step {s} {rec['stage']} {rec['batch_id']} loss "
                                               f"{rec['step_loss']:.4f} lanes {rec['lane_rows']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--mode", choices=["fresh", "resume", "fork"], required=True)
    ap.add_argument("--crash", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    t_start = time.perf_counter()
    setup_determinism()
    ctx = Ctx(a.artifacts, a.config)
    cfg, plan = ctx.cfg, ctx.plan
    last = plan["total_steps"]
    if a.mode == "fresh":
        sched = load_schedule(ctx.artifacts)
        br = BranchRun(ctx, a.run_root, a.branch, a.run_id, sched, attempt=1, last_ckpt="init")
        br.planner = Planner(cfg, plan, sched, ctx.store, br.fw, ctx.tok, a.branch)
        br.ledger = HashChainLedger(br.ldir / "consumption_ledger.jsonl")
        if br.ledger.offset != 0:
            raise SystemExit("a fresh run requires an empty ledger")
        runlog.log(ctx.artifacts, f"[{a.branch}] training started (fresh, run_id={a.run_id}, attempt 1"
                                  f"{', crash armed at step %d' % cfg['crash']['step'] if a.crash else ''})")
        br.run(1, last, cfg["crash"] if a.crash else None)
    elif a.mode == "resume":
        sched = load_schedule(ctx.artifacts)
        ldir = Path(a.run_root) / "ledgers" / a.branch
        ck = list_checkpoints(Path(a.run_root) / "checkpoints", a.branch)[-1]
        meta = json.loads((ck / "meta.json").read_text(encoding="utf-8"))
        pre = read_ledger(ldir / "consumption_ledger.jsonl")
        prev_attempt = max(r["attempt"] for r in pre["records"])
        fdir = ldir / "crash_forensics"
        rep = recover_to_checkpoint(ldir / "consumption_ledger.jsonl", meta["ledger_offset"], meta["ledger_head_hash"], fdir)
        side = [truncate_jsonl_by_step(ldir / n, meta["step"], fdir) for n in
                ("opus_decisions.jsonl", "learning_tokens.csv", "learning_samples.jsonl", "performance_steps.jsonl",
                 "eval_checkpoints.jsonl", "firewall_events.jsonl", "eval_access_log.jsonl")]
        br = BranchRun(ctx, a.run_root, a.branch, a.run_id, sched, attempt=prev_attempt + 1, last_ckpt=meta["checkpoint_id"])
        meta2, dstate = load_checkpoint(ck, br.model, br.opt, br.lrs)
        br.planner = Planner(cfg, plan, sched, ctx.store, br.fw, ctx.tok, a.branch, state=dstate)
        br.ledger = HashChainLedger(ldir / "consumption_ledger.jsonl")
        if br.ledger.offset != meta["ledger_offset"] or br.ledger.head != meta["ledger_head_hash"]:
            raise SystemExit("recovered ledger does not end at the checkpoint's offset/head hash; refusing to resume")
        rep.update({"resumed_from_checkpoint": meta["checkpoint_id"], "checkpoint_step": meta["step"],
                    "attempt": prev_attempt + 1, "side_ledgers": side,
                    "restored": {k: meta2[k] for k in ("model_sha256", "optimizer_sha256", "dataloader_state_sha256")}})
        runlog.log(ctx.artifacts, f"[{a.branch}] run resumed in a new process from {meta['checkpoint_id']} at ledger "
                                  f"offset {meta['ledger_offset']}; orphaned steps {rep['orphaned_records']} moved to "
                                  f"crash_forensics; torn tail {rep['torn_tail_bytes']} bytes")
        first = meta["step"] + 1
        rec = br.step(first)
        got = {"step": first, "batch_id": rec["batch_id"], "batch_hash": rec["batch_hash"],
               "sample_ids": [r["sample_id"] for mb in rec["microbatches"] for r in mb["rows"]],
               "span_ids": [[span_id(s) for s in r["spans"]] for mb in rec["microbatches"] for r in mb["rows"]],
               "row_hashes": [r["row_hash"] for mb in rec["microbatches"] for r in mb["rows"]]}
        pre_rec = [r for r in pre["records"] if r["global_step"] == first]
        exp_ck = meta["next_batch"]
        exp_pre = ({"step": first, "batch_id": pre_rec[0]["batch_id"], "batch_hash": pre_rec[0]["batch_hash"],
                    "sample_ids": [r["sample_id"] for mb in pre_rec[0]["microbatches"] for r in mb["rows"]],
                    "span_ids": [[span_id(s) for s in r["spans"]] for mb in pre_rec[0]["microbatches"] for r in mb["rows"]],
                    "row_hashes": [r["row_hash"] for mb in pre_rec[0]["microbatches"] for r in mb["rows"]]}
                   if pre_rec else None)
        keys = ("batch_id", "batch_hash", "sample_ids", "span_ids", "row_hashes")
        m_ck = all(exp_ck[k] == got[k] for k in keys)
        m_pre = exp_pre is not None and all(exp_pre[k] == got[k] for k in keys)
        rep.update({"first_resumed_step": first, "first_resumed_batch": got,
                    "expected_from_checkpoint_meta": exp_ck, "expected_from_pre_crash_record": exp_pre,
                    "matches_checkpoint_expectation": m_ck, "matches_pre_crash_record": m_pre,
                    "resume_latency_s": time.perf_counter() - t_start})
        runlog.check(ctx.artifacts, "resume_next_batch_matched", m_ck and m_pre,
                     f"step {first}: checkpoint {meta['checkpoint_id']} declared {exp_ck['batch_id']} / "
                     f"{exp_ck['batch_hash'][:16]}; the dead process had written {exp_pre['batch_id'] if exp_pre else None};"
                     f" resumed batch {got['batch_id']} / {got['batch_hash'][:16]} ({len(got['sample_ids'])} samples, "
                     f"ids/spans/row hashes compared)")
        (ldir / "recovery_report.json").write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
        br.run(first + 1, last)
    elif a.mode == "fork":
        fk = cfg["fork"]
        parent, pstep = "main", fk["from_checkpoint_step"]
        pck = Path(a.run_root) / "checkpoints" / parent / f"step_{pstep:05d}"
        pmeta = json.loads((pck / "meta.json").read_text(encoding="utf-8"))
        plpath = Path(a.run_root) / "ledgers" / parent / "consumption_ledger.jsonl"
        pl = read_ledger(plpath, strict=True)["records"]
        if pl[pstep - 1]["record_hash"] != pmeta["ledger_head_hash"]:
            raise SystemExit("parent checkpoint does not match the parent ledger")
        br = BranchRun(ctx, a.run_root, a.branch, a.run_id, None, attempt=1, last_ckpt=pmeta["checkpoint_id"], drills=False)
        ov = {"from_step": pstep + 1, "mix": fk["mix_overrides"]}
        sched = compile_schedule(plan, cfg, ctx.store, br.fw, overrides=ov)
        (br.ldir / "fork_schedule.json").write_text(json.dumps(sched, indent=1, sort_keys=True), encoding="utf-8")
        br.sched = sched
        meta2, dstate = load_checkpoint(pck, br.model, br.opt, br.lrs)
        br.planner = Planner(cfg, plan, sched, ctx.store, br.fw, ctx.tok, a.branch, state=dstate)
        br.ledger = HashChainLedger(br.ldir / "consumption_ledger.jsonl", genesis_hash=pmeta["ledger_head_hash"],
                                    genesis_offset=pmeta["ledger_offset"])
        parent_sched = load_schedule(ctx.artifacts)
        fp = {"fork_branch_id": a.branch, "parent_branch_id": parent, "parent_run_id": pl[0]["run_id"],
              "parent_checkpoint_id": pmeta["checkpoint_id"], "parent_checkpoint_step": pstep,
              "parent_ledger_offset": pmeta["ledger_offset"], "parent_ledger_head_hash": pmeta["ledger_head_hash"],
              "parent_ledger_file_sha256_at_fork": sha256_bytes(plpath.read_bytes()),
              "common_history": {"steps": [1, pstep], "batch_ids_digest": sha256_json([r["batch_id"] for r in pl[:pstep]])},
              "inherited": {"model_sha256": state_hash(br.model.state_dict()), "optimizer_sha256": optimizer_hash(br.opt),
                            "dataloader_state_sha256": sha256_json(dstate), "scheduler_last_epoch": br.lrs.last_epoch},
              "parent_checkpoint_meta": {k: pmeta[k] for k in ("model_sha256", "optimizer_sha256", "dataloader_state_sha256")},
              "parent_declared_next_batch": pmeta["next_batch"]["batch_id"], "reason": fk["reason"],
              "mix_overrides": ov, "parent_schedule_sha256": parent_sched["schedule_sha256"],
              "fork_schedule_sha256": sched["schedule_sha256"], "first_fork_step": pstep + 1,
              "last_fork_step": pstep + fk["steps"]}
        (br.ldir / "fork_point.json").write_text(json.dumps(fp, indent=1, sort_keys=True), encoding="utf-8")
        runlog.log(ctx.artifacts, f"[{a.branch}] branch forked from {pmeta['checkpoint_id']} (parent ledger offset "
                                  f"{pmeta['ledger_offset']}, head {pmeta['ledger_head_hash'][:16]}); new data branch: "
                                  f"{fk['reason']}")
        br.run(pstep + 1, pstep + fk["steps"])


if __name__ == "__main__":
    main(sys.argv[1:])
