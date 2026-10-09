"""Replay an earlier interval of the main branch FROM THE LEDGER (kept from the prototype, adapted).

    python -m tdes.replay --artifacts A --run-root R

The instructor's point about non-determinism: to go back in history we do not re-run the selection code and
hope it agrees, we read the ledger ("that shard was sent, then that one") and send exactly that. So:
  1. ledger-driven rebuild (the replay itself): restore main@s<from> (model, optimizer, LR schedule, dataloader
     state) and, for every historical step, rebuild each row from the ledger's recorded spans + row length and
     the immutable shards; compare batch id, token spans, row/loss-mask/position/attention hashes, batch hash;
     train on the rebuilt rows and compare per-token losses, step loss, grad norm and the model hash after;
  2. cross-check only: the dataloader, restored from the same checkpoint, re-plans each step independently;
     its batch and OPUS decisions should equal the ledger's (this is where silent non-determinism would show);
  3. stream fingerprint: a rolling sha256 over the ordered (batch_id, batch_hash) pairs of the interval,
     original vs replay.
Replay writes its own hash-chained ledger and never touches the main ledgers.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import runlog
from .checkpoint import load_checkpoint
from .context import Ctx, load_schedule, setup_determinism
from .dataloader import Planner
from .firewall import Firewall
from .hashing import sha256_json
from .ledger import HashChainLedger, read_ledger
from .model import optimizer_hash, state_hash
from .packing import batch_identity, materialize_row, row_hashes, span_id
from .trainer import build_model, train_on_microbatches


def stream_fingerprint(pairs):
    h = "0" * 64
    for bid, bh in pairs:
        h = sha256_json([h, bid, bh])
    return h


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    setup_determinism()
    ctx = Ctx(a.artifacts, a.config)
    cfg, plan, store, tok = ctx.cfg, ctx.plan, ctx.store, ctx.tok
    rr = Path(a.run_root)
    s0, s1 = cfg["replay"]["from_checkpoint_step"], cfg["replay"]["to_step"]
    branch = f"replay-main-s{s0}-s{s1}"
    ldir = rr / "ledgers" / branch
    ldir.mkdir(parents=True, exist_ok=True)
    mdir = rr / "ledgers" / "main"
    hist = {r["global_step"]: r for r in read_ledger(mdir / "consumption_ledger.jsonl", strict=True)["records"]
            if s0 < r["global_step"] <= s1}
    main_dec = {}
    for ln in open(mdir / "opus_decisions.jsonl", encoding="utf-8"):
        d = json.loads(ln)
        if s0 < d["step"] <= s1:
            main_dec.setdefault(d["step"], []).append(d)
    main_tok = {}
    with open(mdir / "learning_tokens.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            st = int(row["step"])
            if s0 < st <= s1:
                main_tok.setdefault(st, []).append((int(row["row"]), int(row["target_pos"]), row["loss"]))
    sched = load_schedule(ctx.artifacts)
    fw = Firewall(ctx.artifacts, store, tok, ldir / "firewall_events.jsonl", ldir / "eval_access_log.jsonl")
    model, opt, lrs = build_model(cfg, plan, tok.vocab_size)
    meta, dstate = load_checkpoint(rr / "checkpoints" / "main" / f"step_{s0:05d}", model, opt, lrs)
    planner = Planner(cfg, plan, sched, store, fw, tok, branch, state=dstate)
    ledger = HashChainLedger(ldir / "consumption_ledger.jsonl")
    if ledger.offset:
        raise SystemExit("the replay ledger must start empty")
    runlog.log(ctx.artifacts, f"[replay] restored {meta['checkpoint_id']} (ledger offset {meta['ledger_offset']}); "
                              f"replaying historical steps {s0 + 1}..{s1} from the main ledger")
    steps = []
    for s in range(s0 + 1, s1 + 1):
        h = hist[s]
        # (1) rebuild from the ledger's spans + the immutable shards, microbatch by microbatch
        mbs, hrows = [], []
        for mb in h["microbatches"]:
            cur = []
            for hr in mb["rows"]:
                r = materialize_row(hr["spans"], store, hr["L"], tok.pad_id)
                r.update(row_hashes(r))
                r.update({"spans": hr["spans"], "lane": hr["lane"]})
                cur.append(r)
                hrows.append(hr)
            mbs.append(cur)
        rows = [r for mb in mbs for r in mb]
        ident = batch_identity(s, rows)
        row_cmp = [all(hr[k] == r[k] for k in ("row_hash", "loss_mask_hash", "position_ids_hash", "attention_mask_hash",
                                                "tokens_hash")) for hr, r in zip(hrows, rows)]
        spans_eq = [[span_id(x) for x in hr["spans"]] for hr in hrows] == [[span_id(x) for x in r["spans"]] for r in rows]
        # (2) cross-check: independent re-plan from the restored dataloader state with the replaying model
        mh = state_hash(model.state_dict())
        p = planner.plan_step(s, model, mh, drill_inject=False)
        pid = planner.identify(s, p["rows"])
        dec_main = [(d["candidate_id"], d["status"], d["reason"], d["opus_score"]) for d in main_dec[s]]
        dec_re = [(d["candidate_id"], d["status"], d["reason"], d["opus_score"]) for d in p["decisions"]]
        fw.check_batch(rows, s, branch)
        res = train_on_microbatches(model, opt, lrs, mbs, cfg)
        mine = [(ri, int(i) + 1, repr(float(res["before"][ri][i]))) for ri, r in enumerate(rows)
                for i in np.nonzero(r["loss_mask"])[0]]
        tok_eq = sorted(mine) == sorted(main_tok[s])
        mh_after = state_hash(model.state_dict())
        ledger.append({"branch_id": branch, "replay_of": "main", "global_step": s, "source_ledger_offset": h["ledger_offset"],
                       "source_record_hash": h["record_hash"], "batch_id": ident["batch_id"],
                       "batch_hash": ident["batch_hash"], "batch_loss_mask_hash": ident["batch_loss_mask_hash"],
                       "row_hashes": [r["row_hash"] for r in rows],
                       "span_ids": [[span_id(x) for x in r["spans"]] for r in rows], "step_loss": res["step_loss"],
                       "grad_norm": res["grad_norm"], "lr": res["lr"], "model_sha256_before": mh,
                       "model_sha256_after": mh_after, "token_loss_digest": sha256_json(sorted([list(x) for x in mine])),
                       "replanned_batch_id": pid["batch_id"], "replanned_batch_hash": pid["batch_hash"],
                       "replanned_decisions_digest": sha256_json([list(x) for x in dec_re])})
        steps.append({
            "step": s,
            "original": {k: h[k] for k in ("batch_id", "batch_hash", "batch_loss_mask_hash", "step_loss", "grad_norm",
                                           "model_sha256_after")},
            "replay_from_ledger": {"batch_id": ident["batch_id"], "batch_hash": ident["batch_hash"],
                                   "batch_loss_mask_hash": ident["batch_loss_mask_hash"], "step_loss": res["step_loss"],
                                   "grad_norm": res["grad_norm"], "model_sha256_after": mh_after},
            "replanned_by_dataloader": {"batch_id": pid["batch_id"], "batch_hash": pid["batch_hash"],
                                        "n_decisions": len(dec_re)},
            "checks": {
                "batch_id_match": ident["batch_id"] == h["batch_id"],
                "batch_hash_match": ident["batch_hash"] == h["batch_hash"],
                "loss_mask_hash_match": ident["batch_loss_mask_hash"] == h["batch_loss_mask_hash"],
                "token_spans_match": spans_eq, "all_row_hashes_match": all(row_cmp),
                "token_losses_bitwise_match": tok_eq, "step_loss_match": res["step_loss"] == h["step_loss"],
                "grad_norm_match": res["grad_norm"] == h["grad_norm"],
                "model_hash_after_match": mh_after == h["model_sha256_after"]},
            "cross_check": {"replanned_batch_hash_match": pid["batch_hash"] == h["batch_hash"],
                            "replanned_opus_decisions_match": dec_main == dec_re}})
    end = json.loads((rr / "checkpoints" / "main" / f"step_{s1:05d}" / "meta.json").read_text(encoding="utf-8"))
    final = {"replayed_model_sha256": state_hash(model.state_dict()), "original_checkpoint_id": end["checkpoint_id"],
             "original_model_sha256": end["model_sha256"], "replayed_optimizer_sha256": optimizer_hash(opt),
             "original_optimizer_sha256": end["optimizer_sha256"]}
    final["model_match"] = final["replayed_model_sha256"] == final["original_model_sha256"]
    final["optimizer_match"] = final["replayed_optimizer_sha256"] == final["original_optimizer_sha256"]
    fp = {"original": stream_fingerprint([(hist[s]["batch_id"], hist[s]["batch_hash"]) for s in range(s0 + 1, s1 + 1)]),
          "replay": stream_fingerprint([(x["replay_from_ledger"]["batch_id"], x["replay_from_ledger"]["batch_hash"])
                                        for x in steps])}
    fp["match"] = fp["original"] == fp["replay"]
    ok = (len(steps) == s1 - s0 and all(all(x["checks"].values()) for x in steps) and final["model_match"]
          and final["optimizer_match"] and fp["match"])
    report = {"branch_id": branch, "restored_checkpoint": meta["checkpoint_id"], "restored_ledger_offset": meta["ledger_offset"],
              "interval": [s0 + 1, s1], "steps": steps, "final_state": final, "stream_fingerprint": fp,
              "replay_ledger_head": ledger.head, "all_checks_pass": ok,
              "cross_check_all_match": all(all(x["cross_check"].values()) for x in steps),
              "replay_latency_s": time.perf_counter() - t0,
              "method": "rows rebuilt from ledger spans + immutable shards and retrained from the restored checkpoint; "
                        "the dataloader re-plan is a cross-check, not the source of the replayed stream"}
    (ldir / "replay_report.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    n = len(steps)
    runlog.log(ctx.artifacts, f"historical stream replayed: steps {s0 + 1}..{s1} rebuilt from the ledger; batch id + spans "
                              f"+ hashes matched {sum(all(x['checks'][k] for k in ('batch_id_match', 'batch_hash_match', 'token_spans_match')) for x in steps)}/{n}; "
                              f"token losses bitwise {sum(x['checks']['token_losses_bitwise_match'] for x in steps)}/{n}; "
                              f"stream fingerprint {fp['replay'][:16]} == {fp['original'][:16]}: {fp['match']}; "
                              f"final model == {end['checkpoint_id']}: {final['model_match']}; independent re-plan agrees: "
                              f"{report['cross_check_all_match']}")
    runlog.check(ctx.artifacts, "replay_hash_matched", ok, f"{n} steps, replay ledger head {ledger.head[:16]}")


if __name__ == "__main__":
    main(sys.argv[1:])
