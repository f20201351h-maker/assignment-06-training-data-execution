"""Data audit: answer questions about what trained which checkpoint, from the ledgers alone.

    python -m tdes.audit --artifacts A

Writes ledgers/audit_report.json and ledgers/learning_shards.json (the shard learning report card).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from . import runlog
from .ledger import read_ledger


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")] if Path(p).exists() else []


def influence(records, lo, hi):
    shards, lanes, docs = defaultdict(int), defaultdict(int), set()
    for r in records:
        if lo <= r["global_step"] <= hi:
            for mb in r["microbatches"]:
                for row in mb["rows"]:
                    lanes[row["lane"]] += row["n_loss_tokens"]
                    for sp in row["spans"]:
                        shards[sp["shard_id"]] += sp["end"] - sp["start"]
                        docs.add(sp["doc_id"])
    return {"steps": [lo, hi], "tokens_by_shard": dict(sorted(shards.items())),
            "loss_tokens_by_lane": dict(sorted(lanes.items())), "n_documents": len(docs)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    a = ap.parse_args(argv)
    art = Path(a.artifacts)
    ld = art / "ledgers" / "main"
    recs = read_ledger(ld / "consumption_ledger.jsonl", strict=True)["records"]
    decs = jl(ld / "opus_decisions.jsonl")
    samples = jl(ld / "learning_samples.jsonl")
    metas = [json.loads((c / "meta.json").read_text(encoding="utf-8"))
             for c in sorted((art / "checkpoints" / "main").iterdir()) if (c / "meta.json").exists()]
    rep = {"source": "ledgers/main/consumption_ledger.jsonl (hash chain verified on read)"}
    rep["q1_data_behind_each_checkpoint"] = [{"checkpoint_id": m["checkpoint_id"], "ledger_offset": m["ledger_offset"],
                                              **influence(recs, 1, m["step"])} for m in metas]
    # "which shards influenced the model between X and Y tokens?" on the cumulative position counter
    cum, lo_tok, hi_tok = 0, metas[-3]["tokens_seen"], metas[-2]["tokens_seen"]
    win = []
    for r in recs:
        start, cum = cum, cum + r["token_positions"]
        if start >= lo_tok and cum <= hi_tok:
            win.append(r["global_step"])
    q2 = influence(recs, win[0], win[-1])
    q2["token_window"] = [lo_tok, hi_tok]
    rep["q2_token_window"] = q2
    jumps = [(recs[i]["step_loss"] - recs[i - 1]["step_loss"], recs[i]["global_step"]) for i in range(1, len(recs))]
    dj, spike = max(jumps)
    rep["q3_largest_loss_increase"] = {
        "step": spike, "loss_increase": dj, "stage": recs[spike - 1]["stage"],
        "preceding_accepted_opus_decisions": [{k: d[k] for k in ("decision_id", "step", "lane", "reason", "opus_score",
                                                                 "shard_ids")}
                                              for d in decs if spike - 2 <= d["step"] <= spike and d["status"] == "accepted"]}
    fate = defaultdict(list)
    for d in decs:
        fate[d["candidate_id"]].append(d["status"])
    trained = {row["candidate_id"] for r in recs for mb in r["microbatches"] for row in mb["rows"]}
    by_lane = defaultdict(lambda: defaultdict(int))
    for d in decs:
        by_lane[d["lane"]][f"{d['status']}:{d['reason']}"] += 1
    rep["q4_opus_review_queue"] = {
        "candidates_considered": len(fate), "candidates_trained": len(trained & set(fate)),
        "candidates_never_trained": sum(1 for c in fate if c not in trained),
        "deferred_then_accepted": sum(1 for st in fate.values() if "deferred" in st and st[-1] == "accepted"),
        "decisions_by_lane": {k: dict(v) for k, v in sorted(by_lane.items())},
        "note": "every scored candidate keeps its spans in opus_decisions.jsonl, so rejected clean data can be "
                "reconsidered for the anneal or a later model"}
    # what the floor rescued: protected rows the selector would have rejected, and how surprising they were
    over = {d["candidate_id"] for d in decs if d["protected_floor_override"]}
    ov = [s for s in samples if s["accept_reason"] == "protected_floor_override" and s["n_loss_tokens"]]
    rest = [s for s in samples if s["accept_reason"] != "protected_floor_override" and s["n_loss_tokens"]]
    mean = lambda xs: sum(x["loss_sum"] for x in xs) / max(1, sum(x["n_loss_tokens"] for x in xs))
    rep["q5_protected_floor_rescues"] = {
        "override_decisions": len(over), "by_lane": dict(sorted(defaultdict(int, {
            l: sum(1 for d in decs if d["protected_floor_override"] and d["lane"] == l) for l in by_lane}).items())),
        "mean_token_loss_overridden_rows": mean(ov), "mean_token_loss_other_rows": mean(rest),
        "reading": "rows OPUS would have rejected but the floor kept; a higher loss here means the English proxy "
                   "is undervaluing data the model has not learned yet (the proxy-bias signal)"}
    card = defaultdict(lambda: {"n_samples": 0, "loss_tokens": 0, "loss_sum": 0.0, "loss_after_sum": 0.0,
                                "scores": [], "by_pass": defaultdict(lambda: [0, 0.0]), "classes": defaultdict(int),
                                "lane": None, "phases": defaultdict(int), "overrides": 0})
    for s in samples:
        c = card[s["shard_id"]]
        c["lane"] = s["lane"]
        c["n_samples"] += 1
        c["loss_tokens"] += s["n_loss_tokens"]
        c["loss_sum"] += s["loss_sum"]
        c["loss_after_sum"] += s["loss_after_sum"]
        c["scores"].append(s["opus_score"])
        c["by_pass"][s["repeat_pass"]][0] += s["n_loss_tokens"]
        c["by_pass"][s["repeat_pass"]][1] += s["loss_sum"]
        c["classes"][s["classification"]] += 1
        c["phases"][s["phase"]] += 1
        c["overrides"] += s["accept_reason"] == "protected_floor_override"
    shards = {}
    for sid, c in sorted(card.items()):
        n = c["loss_tokens"] or 1
        delta = (c["loss_after_sum"] - c["loss_sum"]) / n
        shards[sid] = {"lane": c["lane"], "n_samples": c["n_samples"], "loss_tokens": c["loss_tokens"],
                       "mean_token_loss": c["loss_sum"] / n, "mean_loss_delta_after_update": delta,
                       "mean_opus_score": sum(c["scores"]) / len(c["scores"]), "floor_overrides": c["overrides"],
                       "mean_loss_by_repeat_pass": {str(p): v[1] / v[0] for p, v in sorted(c["by_pass"].items()) if v[0]},
                       "sample_classifications": dict(c["classes"]), "phases_seen": dict(c["phases"]),
                       "classification": "useful" if delta < -0.05 else ("harmful" if delta > 0.05 else "neutral")}
    (art / "ledgers" / "learning_shards.json").write_text(json.dumps(shards, indent=1, sort_keys=True), encoding="utf-8")
    rep["q6_shard_report_card"] = "ledgers/learning_shards.json"
    rep["q7_eval_firewall_trace"] = {
        "blocked_events": jl(art / "ledgers" / "build" / "firewall_events.jsonl") + jl(ld / "firewall_events.jsonl"),
        "eval_reads": jl(ld / "eval_access_log.jsonl")}
    (art / "ledgers" / "audit_report.json").write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    runlog.log(art, f"audit completed: {len(metas)} checkpoints traced to their data; token window {lo_tok}-{hi_tok} "
                    f"(steps {win[0]}-{win[-1]}) touched {len(q2['tokens_by_shard'])} shards; largest loss increase at "
                    f"step {spike}; {rep['q4_opus_review_queue']['candidates_never_trained']} considered-but-untrained "
                    f"candidates kept for review; {len(over)} floor rescues (mean loss {mean(ov):.3f} vs {mean(rest):.3f}); "
                    f"{len(shards)} shard report cards")


if __name__ == "__main__":
    main(sys.argv[1:])
