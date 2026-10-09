"""One command for the whole demonstration:

    python run_demo.py

Deletes and regenerates artifacts/. Every phase runs in its own Python process, so a later phase
can only use what an earlier phase wrote to disk:

  1. build      vendored A2/A4/A5 inputs -> ingress gate -> shards -> manifests -> eval registry -> A5 schedule
  2. train      main branch from scratch; the process kills itself while committing the crash step's ledger
                record (torn write + os._exit)
  3. inspect    this process records what the dead process left on disk
  4. resume     a new process recovers the ledger to the last checkpoint and continues to the end
  5. reference  the same run without a crash, in a separate folder (control for resume / determinism)
  6. replay     restore an earlier checkpoint and replay that interval from the ledger
  7. fork       restore a checkpoint and start a new data branch (A5 proxy arm H1)
  8. audit, performance, verification -> evidence.json / evidence.md
Exit code 0 only if the verifier marks every requirement PASS. Nothing here reads the upstream assignment
folders; everything comes from inputs/ in this repository.
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from tdes import runlog  # noqa: E402
from tdes.checkpoint import list_checkpoints  # noqa: E402
from tdes.hashing import sha256_json  # noqa: E402
from tdes.ledger import read_ledger  # noqa: E402


def _rm_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def child(art, *args, expect=0):
    t = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", *args], cwd=REPO)
    if p.returncode != expect:
        runlog.log(art, f"phase {args[0]} exited with {p.returncode}, expected {expect}; stopping")
        sys.exit(1)
    return p.returncode, time.perf_counter() - t


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Run the complete Training Data Execution System demonstration.")
    ap.add_argument("--artifacts", default=str(REPO / "artifacts"))
    ap.add_argument("--config", default=str(REPO / "config" / "demo_config.json"))
    a = ap.parse_args(argv)
    art = Path(a.artifacts).resolve()
    C = ["--config", str(Path(a.config).resolve())]
    if art.exists():
        shutil.rmtree(art, onexc=_rm_readonly) if sys.version_info >= (3, 12) else shutil.rmtree(art, onerror=_rm_readonly)
    art.mkdir(parents=True)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    A = str(art)
    import platform
    from importlib.metadata import version
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
           "torch": version("torch"), "numpy": version("numpy"), "tokenizers": version("tokenizers"),
           "pyyaml": version("pyyaml"), "torch_threads": 1, "device": "cpu", "config": Path(a.config).name,
           "config_sha256": sha256_json(cfg)}
    (art / "environment.json").write_text(json.dumps(env, indent=1, sort_keys=True), encoding="utf-8")
    runlog.log(art, f"demo started: python {env['python']}, torch {env['torch']}, tokenizers {env['tokenizers']}, "
                    f"{env['platform']}; artifacts -> {art.name}/")
    child(art, "tdes.build", "--artifacts", A, *C)
    reg = json.loads((art / "manifests" / "shard_registry.json").read_text(encoding="utf-8"))
    run_id = "run-" + sha256_json({"config": cfg, "registry": reg["registry_sha256"],
                                   "tokenizer": cfg["tokenizer"]["tokenizer_id"]})[:12]
    runlog.log(art, f"run_id {run_id}")
    code, _ = child(art, "tdes.trainer", "--artifacts", A, "--run-root", A, "--branch", "main", "--run-id", run_id,
                    "--mode", "fresh", "--crash", *C, expect=cfg["crash"]["exit_code"])
    ldir = art / "ledgers" / "main"
    st = read_ledger(ldir / "consumption_ledger.jsonl")
    cks = list_checkpoints(art / "checkpoints", "main")
    last = json.loads((cks[-1] / "meta.json").read_text(encoding="utf-8"))
    crash = {"child_exit_code": code, "expected_exit_code": cfg["crash"]["exit_code"],
             "configured_crash_step": cfg["crash"]["step"], "crash_point": cfg["crash"]["point"],
             "committed_ledger_records": st["offset"], "committed_steps": [r["global_step"] for r in st["records"]],
             "torn_tail_bytes": st["torn_tail_bytes"], "torn_tail_preview": st["torn_tail_preview"],
             "chain_errors": st["errors"], "checkpoints_on_disk": [c.name for c in cks],
             "latest_checkpoint": last["checkpoint_id"], "latest_checkpoint_ledger_offset": last["ledger_offset"],
             "steps_committed_after_latest_checkpoint": [r["global_step"] for r in st["records"] if r["global_step"] > last["step"]],
             "expected_next_step": last["step"] + 1, "expected_next_batch_declared_by_checkpoint": last["next_batch"]}
    (art / "ledgers" / "crash_report.json").write_text(json.dumps(crash, indent=1, sort_keys=True), encoding="utf-8")
    runlog.log(art, f"crash simulated: the training process died with exit code {code} during step {cfg['crash']['step']}; "
                    f"ledger has {st['offset']} committed records + {st['torn_tail_bytes']} torn bytes; latest checkpoint "
                    f"{last['checkpoint_id']} (ledger offset {last['ledger_offset']}) says the next batch must be "
                    f"{last['next_batch']['batch_id']}")
    child(art, "tdes.trainer", "--artifacts", A, "--run-root", A, "--branch", "main", "--run-id", run_id, "--mode", "resume", *C)
    runlog.log(art, "reference run: same config, uninterrupted, separate run root (control for resume and determinism)")
    child(art, "tdes.trainer", "--artifacts", A, "--run-root", str(art / "reference_run"), "--branch",
          "reference-uninterrupted", "--run-id", run_id + "-reference", "--mode", "fresh", *C)
    child(art, "tdes.replay", "--artifacts", A, "--run-root", A, *C)
    child(art, "tdes.trainer", "--artifacts", A, "--run-root", A, "--branch", cfg["fork"]["branch_id"], "--run-id", run_id,
          "--mode", "fork", *C)
    child(art, "tdes.audit", "--artifacts", A)
    child(art, "tdes.perf", "--artifacts", A, *C)
    sys.exit(subprocess.run([sys.executable, "-m", "tdes.verify", "--artifacts", A, *C], cwd=REPO).returncode)


if __name__ == "__main__":
    main(sys.argv[1:])
