"""Compare two independently generated artifacts/ folders on everything that should be identical.

    python tools/compare_runs.py RUN_A RUN_B

Compared: shard registry and every shard manifest, the compiled schedule, every consumption-ledger record of
main / reference / fork / replay (minus nothing: records carry no wall-clock fields), OPUS decisions, the
token-level learning trace, checkpoint meta (model / optimizer / dataloader hashes, ledger binding, next
batch), and the per-requirement results in evidence.json. Excluded on purpose: run.log (timestamps, pids),
performance.json and performance_steps.jsonl (wall-clock), environment.json, evidence generated_at, and
the latency fields of the recovery / replay reports. Exit code 0 only if everything compared is equal.
"""
import hashlib
import json
import sys
from pathlib import Path


def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a, b):
    a, b = Path(a), Path(b)
    rel = ["manifests/shard_registry.json", "manifests/mixture_schedule.json", "manifests/a5_plan.json",
           "manifests/eval_registry.json", "manifests/admission_report.json", "upstream_contracts.json",
           "ledgers/main/consumption_ledger.jsonl", "ledgers/main/opus_decisions.jsonl",
           "ledgers/main/learning_tokens.csv", "ledgers/main/learning_samples.jsonl"]
    rel += [str(p.relative_to(a)).replace("\\", "/") for p in sorted((a / "manifests" / "shards").glob("*.json"))]
    rel += [str(p.relative_to(a)).replace("\\", "/") for p in sorted((a / "ledgers").glob("*/consumption_ledger.jsonl"))]
    rel += ["reference_run/ledgers/reference-uninterrupted/consumption_ledger.jsonl"]
    rel += [str(p.relative_to(a)).replace("\\", "/") for p in sorted((a / "checkpoints").glob("*/*/meta.json"))]
    rel += [str(p.relative_to(a)).replace("\\", "/") for p in sorted((a / "checkpoints").glob("*/*/model.pt"))]
    diff, same = [], 0
    for r in sorted(set(rel)):
        pa, pb = a / r, b / r
        if not pb.exists():
            diff.append(f"missing in B: {r}")
        elif h(pa) != h(pb):
            diff.append(f"differs: {r}")
        else:
            same += 1
    ea, eb = json.loads((a / "evidence.json").read_text(encoding="utf-8")), json.loads((b / "evidence.json").read_text(encoding="utf-8"))
    ra = {q["id"]: (q["result"], [c["pass"] for c in q["checks"]]) for q in ea["requirements"]}
    rb = {q["id"]: (q["result"], [c["pass"] for c in q["checks"]]) for q in eb["requirements"]}
    if ra != rb or ea["overall"] != eb["overall"]:
        diff.append("evidence results differ")
    for name in ("ledgers/main/recovery_report.json",):
        xa, xb = (json.loads((x / name).read_text(encoding="utf-8")) for x in (a, b))
        for k in ("resume_latency_s",):
            xa.pop(k, None)
            xb.pop(k, None)
        xa.pop("forensic_copy", None)
        xb.pop("forensic_copy", None)
        if xa != xb:
            diff.append(f"differs: {name} (excluding latency and paths)")
        else:
            same += 1
    out = {"run_a": str(a), "run_b": str(b), "files_identical": same, "differences": diff,
           "evidence_overall": [ea["overall"], eb["overall"]], "identical": not diff}
    print(json.dumps(out, indent=1))
    return 0 if not diff else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
