"""Performance report built only from raw per-step records (adapted from the prototype).

    python -m tdes.perf --artifacts A

Every number in performance.json is a sum or ratio of fields in ledgers/<branch>/performance_steps.jsonl
(timings) and the consumption ledger (token counts); the formulas and raw totals are listed next to the
numbers so the verifier, or a reader, can recompute them. CPU only: GPU metrics are reported as null with a
reason rather than as zeros.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from . import runlog
from .context import load_config, load_tokenizer
from .ledger import read_ledger
from .packing import rows_needed
from .shards import ShardStore

FORMULAS = {
    "packing_utilization": "sum(n_real_tokens) / sum(token_positions)",
    "loss_bearing_fraction": "sum(n_loss_tokens) / sum(token_positions)",
    "padding_fraction": "1 - sum(n_real_tokens) / sum(token_positions)",
    "context_only_fraction": "(sum(n_real_tokens) - sum(n_loss_tokens)) / sum(token_positions)",
    "raw_token_positions_per_s": "sum(token_positions) / sum(t_step_total_s)",
    "useful_loss_bearing_tokens_per_s": "sum(n_loss_tokens) / sum(t_step_total_s)",
    "accepted_real_tokens_per_s": "sum(n_real_tokens) / sum(t_step_total_s)  (rows that passed OPUS)",
    "candidate_positions_scored_per_s": "sum(candidate_positions_scored) / sum(t_step_total_s)",
    "loader_time_fraction": "sum(t_plan_and_score_s) / sum(t_step_total_s)",
    "opus_acceptance_rate": "accepted rows / candidates scored (all lanes, including always-accepted ones)",
    "web_selector_acceptance_rate": "accepted / scored, web lane while the selector is active (A5 keep 0.40)",
    "useful_loss_bearing_tokens_per_s_end_to_end": "sum(n_loss_tokens) / (sum(t_step_total_s) + sum(t_checkpoint_s) "
                                                   "+ time of steps lost in the crash)",
}


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")] if Path(p).exists() else []


def branch_perf(art, ldir):
    fpp = ldir / "fork_point.json"
    kw = {}
    if fpp.exists():
        fp = json.loads(fpp.read_text(encoding="utf-8"))
        kw = {"genesis_hash": fp["parent_ledger_head_hash"], "genesis_offset": fp["parent_ledger_offset"]}
    recs = read_ledger(ldir / "consumption_ledger.jsonl", strict=True, **kw)["records"]
    perf = {p["step"]: p for p in jl(ldir / "performance_steps.jsonl")}
    steps = [r["global_step"] for r in recs]
    S = lambda k: sum(perf[s][k] for s in steps)
    raw = {"steps": len(steps), "token_positions": sum(r["token_positions"] for r in recs),
           "n_real_tokens": sum(r["n_real_tokens"] for r in recs), "n_loss_tokens": sum(r["n_loss_tokens"] for r in recs),
           "t_step_total_s": S("t_step_total_s"), "t_plan_and_score_s": S("t_plan_and_score_s"), "t_train_s": S("t_train_s"),
           "t_firewall_s": S("t_firewall_s"), "t_ledger_s": S("t_ledger_s"), "t_checkpoint_s": S("t_checkpoint_s"),
           "candidates_scored": S("candidates_scored"), "candidate_positions_scored": S("candidate_positions_scored"),
           "accepted_rows": S("accepted_rows")}
    T = raw["t_step_total_s"]
    m = {"packing_utilization": raw["n_real_tokens"] / raw["token_positions"],
         "loss_bearing_fraction": raw["n_loss_tokens"] / raw["token_positions"],
         "padding_fraction": 1 - raw["n_real_tokens"] / raw["token_positions"],
         "context_only_fraction": (raw["n_real_tokens"] - raw["n_loss_tokens"]) / raw["token_positions"],
         "raw_token_positions_per_s": raw["token_positions"] / T, "useful_loss_bearing_tokens_per_s": raw["n_loss_tokens"] / T,
         "accepted_real_tokens_per_s": raw["n_real_tokens"] / T,
         "candidate_positions_scored_per_s": raw["candidate_positions_scored"] / T,
         "loader_time_fraction": raw["t_plan_and_score_s"] / T,
         "opus_acceptance_rate": raw["accepted_rows"] / raw["candidates_scored"]}
    web = [d for d in jl(ldir / "opus_decisions.jsonl") if d["lane_mode"] == "selector" and d["opus_score"] is not None]
    raw["web_selector_scored"], raw["web_selector_accepted"] = len(web), sum(d["status"] == "accepted" for d in web)
    m["web_selector_acceptance_rate"] = raw["web_selector_accepted"] / max(1, raw["web_selector_scored"])
    lost = [p for p in jl(ldir / "crash_forensics" / "performance_steps.jsonl.pre_crash")
            if p["step"] not in perf or p["attempt"] != perf[p["step"]]["attempt"]]
    raw["t_lost_work_s"] = sum(p["t_step_total_s"] for p in lost)
    raw["lost_steps"] = [p["step"] for p in lost]
    raw["t_end_to_end_s"] = T + raw["t_checkpoint_s"] + raw["t_lost_work_s"]
    m["useful_loss_bearing_tokens_per_s_end_to_end"] = raw["n_loss_tokens"] / raw["t_end_to_end_s"]
    man = {}

    def doc_len(sid, did):
        if sid not in man:
            man[sid] = {d["doc_id"]: d["length"] for d in json.loads(
                (art / "manifests" / "shards" / f"{sid}.json").read_text(encoding="utf-8"))["doc_index"]}
        return man[sid][did]
    pol = defaultdict(lambda: {"rows": 0, "positions": 0, "real": 0, "loss": 0, "docs": {}, "L": set()})
    for r in recs:
        for mb in r["microbatches"]:
            for row in mb["rows"]:
                p = pol[f"{row['policy']}@{row['L']}"]
                p["rows"] += 1
                p["positions"] += row["L"]
                p["real"] += row["n_real_tokens"]
                p["loss"] += row["n_loss_tokens"]
                p["L"].add(row["L"])
                for sp in row["spans"]:
                    p["docs"][(sp["shard_id"], sp["doc_id"], sp["pass"])] = doc_len(sp["shard_id"], sp["doc_id"])
    by_policy = {}
    for k, p in sorted(pol.items()):
        L = int(k.split("@")[1])
        lens = list(p["docs"].values())
        alt = {a: rows_needed(lens, L, a) for a in ("pad_only", "concat_chop", "next_fit", "best_fit_decreasing")}
        by_policy[k] = {"rows": p["rows"], "row_len": L, "token_positions": p["positions"], "real_tokens": p["real"],
                        "loss_tokens": p["loss"], "utilization": p["real"] / p["positions"],
                        "loss_bearing_fraction": p["loss"] / p["positions"], "consumed_documents": len(lens),
                        "consumed_document_tokens": sum(lens),
                        "counterfactual_rows_for_same_documents": alt,
                        "counterfactual_utilization_for_same_documents": {a: sum(lens) / (n * L) for a, n in alt.items()}}
    return {"raw_totals": raw, "metrics": m, "by_policy": by_policy, "steps": [steps[0], steps[-1]]}


def shard_read_latency(art, cfg):
    """Cold open of every training shard: manifest check + read + content-hash verification."""
    tok = load_tokenizer(cfg)
    store = ShardStore(art, tok.tokenizer_id)
    out = []
    for sid in store.shards_where(split="train"):
        t = time.perf_counter()
        store.arrays(sid)
        out.append(time.perf_counter() - t)
    out.sort()
    return {"shards": len(out), "mean_s": sum(out) / len(out), "p50_s": out[len(out) // 2], "max_s": out[-1],
            "definition": "ShardStore.arrays(): manifest hash check, read both arrays, recompute the content hash"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    art = Path(a.artifacts)
    cfg = load_config(a.config)
    out = {"hardware": "CPU, torch.set_num_threads(1), deterministic algorithms", "formulas": FORMULAS,
           "timing_definition": "t_step_total_s = one training step from planning (candidate packing + OPUS scoring) "
                                "through firewall, forward/backward/optimizer and ledger writes. Checkpointing "
                                "(validation, next-batch preview, read-back) is t_checkpoint_s, only in the end-to-end metric",
           "gpu_metrics": {"gpu_idle_time": None, "loader_wait_time_on_gpu": None,
                           "reason": "CPU-only demo: there is no accelerator, so these are not measured (not zero)"},
           "branches": {}}
    out["branches"]["main"] = branch_perf(art, art / "ledgers" / "main")
    out["branches"]["reference-uninterrupted"] = branch_perf(art, art / "reference_run" / "ledgers" / "reference-uninterrupted")
    out["branches"][cfg["fork"]["branch_id"]] = branch_perf(art, art / "ledgers" / cfg["fork"]["branch_id"])
    rec = json.loads((art / "ledgers" / "main" / "recovery_report.json").read_text(encoding="utf-8"))
    rp = json.loads((art / "ledgers" / f"replay-main-s{cfg['replay']['from_checkpoint_step']}-s{cfg['replay']['to_step']}"
                     / "replay_report.json").read_text(encoding="utf-8"))
    lanes = defaultdict(lambda: defaultdict(int))
    for d in jl(art / "ledgers" / "main" / "opus_decisions.jsonl"):
        lanes[d["lane"]][d["status"]] += 1
        lanes[d["lane"]]["protected_floor_override"] += d["protected_floor_override"]
    out["opus_by_lane"] = {l: {"decisions": sum(v[k] for k in ("accepted", "rejected", "deferred")), **v,
                               "rejection_rate": v["rejected"] / sum(v[k] for k in ("accepted", "rejected", "deferred"))}
                           for l, v in sorted(lanes.items())}
    out["latency"] = {"resume_latency_s": rec["resume_latency_s"],
                      "resume_latency_definition": "resume process start (imports included) -> first resumed step committed",
                      "replay_latency_s": rp["replay_latency_s"], "replay_steps": len(rp["steps"]),
                      "shard_read": shard_read_latency(art, cfg)}
    out["headline"] = {k: out["branches"]["main"]["metrics"][k] for k in
                       ("packing_utilization", "loss_bearing_fraction", "padding_fraction", "context_only_fraction",
                        "useful_loss_bearing_tokens_per_s", "accepted_real_tokens_per_s", "raw_token_positions_per_s",
                        "opus_acceptance_rate", "web_selector_acceptance_rate",
                        "useful_loss_bearing_tokens_per_s_end_to_end")}
    (art / "performance.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    h = out["headline"]
    runlog.log(art, f"performance measured: packing utilization {h['packing_utilization']:.4f}, loss-bearing fraction "
                    f"{h['loss_bearing_fraction']:.4f}, useful loss-bearing tokens/s {h['useful_loss_bearing_tokens_per_s']:.1f}"
                    f" (end to end {h['useful_loss_bearing_tokens_per_s_end_to_end']:.1f}), raw positions/s "
                    f"{h['raw_token_positions_per_s']:.1f}, OPUS acceptance {h['opus_acceptance_rate']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
