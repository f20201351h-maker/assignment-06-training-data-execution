"""End-to-end: run the real run_demo.py (12-step config, every phase in its own process), then attack its artifacts.

Each tamper test copies the generated artifacts, breaks one thing and requires the verifier to turn the
matching requirement into FAIL. Several attacks re-chain the ledger afterwards (a forger who knows the hash
format), so the failure has to come from re-computation, not from the chain alone.
"""
import json

import pytest

from conftest import copytree, run_verify
from tdes.hashing import canonical_json, sha256_bytes
from tdes.ledger import read_ledger

TABLE = ["Tokenizer integrity", "Evaluation firewall", "Packing correctness", "Mixture compliance", "OPUS audit trail",
         "Crash recovery", "Replay", "Learning trace", "Throughput"]


def test_demo_pipeline_passes(demo_small):
    assert demo_small["returncode"] == 0, demo_small["stdout"][-3000:] + demo_small["stderr"][-3000:]
    ev = json.loads((demo_small["art"] / "evidence.json").read_text(encoding="utf-8"))
    assert ev["overall"] == "PASS"
    assert [q["table_name"] for q in ev["requirements"][:2]] == ["Tokenizer integrity", "Shards, manifests & immutability"]
    assert set(TABLE) <= {q["table_name"] for q in ev["requirements"]}
    for p in ("run.log", "evidence.json", "evidence.md", "manifests", "ledgers", "checkpoints", "performance.json"):
        assert (demo_small["art"] / p).exists()


def test_crash_was_real_and_resume_is_exact(demo_small):
    art = demo_small["art"]
    cr = json.loads((art / "ledgers" / "crash_report.json").read_text())
    assert cr["child_exit_code"] == 137 and cr["torn_tail_bytes"] > 0 and cr["steps_committed_after_latest_checkpoint"] == [7]
    rep = json.loads((art / "ledgers" / "main" / "recovery_report.json").read_text())
    assert rep["matches_checkpoint_expectation"] and rep["matches_pre_crash_record"]
    main = read_ledger(art / "ledgers" / "main" / "consumption_ledger.jsonl", strict=True)["records"]
    ref = read_ledger(art / "reference_run" / "ledgers" / "reference-uninterrupted" / "consumption_ledger.jsonl",
                      strict=True)["records"]
    assert [r["global_step"] for r in main] == list(range(1, 13))
    assert [(r["batch_id"], r["batch_hash"], r["model_sha256_after"]) for r in main] == \
           [(r["batch_id"], r["batch_hash"], r["model_sha256_after"]) for r in ref]


@pytest.fixture()
def tampered(demo_small, tmp_path):
    assert demo_small["returncode"] == 0
    art = tmp_path / "art"
    copytree(demo_small["art"], art)
    return art, demo_small["config"]


def _rechain(path, recs, genesis="0" * 64):
    prev, out = genesis, []
    for i, r in enumerate(recs, 1):
        r = dict(r, ledger_offset=r["ledger_offset"] if "ledger_offset" in r and genesis != "0" * 64 else i,
                 prev_record_hash=prev)
        r.pop("record_hash", None)
        r["record_hash"] = sha256_bytes(canonical_json(r))
        prev = r["record_hash"]
        out.append(json.dumps(r, sort_keys=True, ensure_ascii=False))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _main(art):
    p = art / "ledgers" / "main" / "consumption_ledger.jsonl"
    return p, [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


def test_edited_record_breaks_the_chain(tampered):
    art, cfg = tampered
    p, recs = _main(art)
    recs[2]["step_loss"] += 0.1
    p.write_text("\n".join(json.dumps(r, sort_keys=True) for r in recs) + "\n", encoding="utf-8")
    rc, req = run_verify(art, cfg)
    assert rc != 0 and req["R11"]["result"] == "FAIL"


def test_rechained_history_with_eval_span_fails_firewall(tampered):
    art, cfg = tampered
    p, recs = _main(art)
    reg = json.loads((art / "manifests" / "shard_registry.json").read_text())
    val = next(e["shard_id"] for e in reg["shards"] if e["split"] == "validation")
    doc = json.loads((art / "manifests" / "shards" / f"{val}.json").read_text(encoding="utf-8"))["doc_index"][0]["doc_id"]
    recs[4]["microbatches"][0]["rows"][0]["spans"] = [{"shard_id": val, "doc_id": doc, "start": 0, "end": 40, "pass": 1}]
    _rechain(p, recs)
    rc, req = run_verify(art, cfg)
    assert rc != 0 and req["R02"]["result"] == "FAIL"


def test_repeated_batch_fails_crash_recovery(tampered):
    art, cfg = tampered
    p, recs = _main(art)
    recs[7] = dict(recs[6], global_step=8)  # step 8 silently re-serves step 7's batch
    _rechain(p, recs)
    rc, req = run_verify(art, cfg)
    assert rc != 0 and req["R06"]["result"] == "FAIL"


def test_relabelled_lane_fails_mixture(tampered):
    art, cfg = tampered
    p, recs = _main(art)
    row = next(r for r in recs[1]["microbatches"][0]["rows"] if r["lane"] == "web")
    row["lane"] = "indic"
    _rechain(p, recs)
    rc, req = run_verify(art, cfg)
    assert rc != 0 and req["R04"]["result"] == "FAIL"


def test_faked_token_loss_fails_learning_trace(tampered):
    art, cfg = tampered
    p = art / "ledgers" / "main" / "learning_tokens.csv"
    lines = p.read_text(encoding="utf-8").split("\n")
    i = lines[0].split(",").index("loss")
    cols = lines[5].split(",")
    cols[i] = repr(float(cols[i]) + 1.0)
    lines[5] = ",".join(cols)
    p.write_text("\n".join(lines), encoding="utf-8")
    rc, req = run_verify(art, cfg)
    assert req["R08"]["result"] == "FAIL"


def test_flipped_opus_decision_fails(tampered):
    art, cfg = tampered
    p = art / "ledgers" / "main" / "opus_decisions.jsonl"
    ds = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    d = next(d for d in ds if d["reason"] == "low_proxy_utility")
    d["status"], d["reason"] = "deferred", "marginal_utility"
    p.write_text("\n".join(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in ds) + "\n", encoding="utf-8")
    rc, req = run_verify(art, cfg)
    assert req["R05"]["result"] == "FAIL"


def test_doctored_replay_fails(tampered):
    art, cfg = tampered
    p = next((art / "ledgers").glob("replay-main-*")) / "consumption_ledger.jsonl"
    recs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    recs[1]["batch_hash"] = "0" * 64
    _rechain(p, recs)
    rc, req = run_verify(art, cfg)
    assert req["R07"]["result"] == "FAIL"


def test_inflated_throughput_fails(tampered):
    art, cfg = tampered
    p = art / "performance.json"
    perf = json.loads(p.read_text())
    perf["branches"]["main"]["metrics"]["useful_loss_bearing_tokens_per_s"] *= 3
    p.write_text(json.dumps(perf))
    rc, req = run_verify(art, cfg)
    assert req["R09"]["result"] == "FAIL"


def test_corrupted_checkpoint_fails_binding(tampered):
    art, cfg = tampered
    f = sorted((art / "checkpoints" / "main").iterdir())[0] / "optimizer.pt"
    b = bytearray(f.read_bytes())
    b[-10] ^= 0xFF
    f.write_bytes(bytes(b))
    rc, req = run_verify(art, cfg)
    assert req["R12"]["result"] == "FAIL"


def test_missing_log_event_fails_end_to_end(tampered):
    art, cfg = tampered
    p = art / "run.log"
    p.write_text(p.read_text(encoding="utf-8").replace("historical stream replayed", "replay done"), encoding="utf-8")
    rc, req = run_verify(art, cfg)
    assert req["R15"]["result"] == "FAIL"
