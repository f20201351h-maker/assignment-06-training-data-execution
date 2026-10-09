import copy
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tdes.context import load_config, load_tokenizer  # noqa: E402


def _rm(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def copytree(src, dst):
    shutil.copytree(src, dst)
    for root, _, files in os.walk(dst):
        for f in files:
            os.chmod(os.path.join(root, f), stat.S_IWRITE | stat.S_IREAD)


def small_config():
    """Same corpus, tokenizer, A5 contract and policies; 10 main steps (S1 1-2, S2 3-5, S3 6-8, S4 9-10) + a
    2-step anneal. Checkpoints every 3 steps; crash at step 8 (checkpoint 6, orphan step 7); replay 3->6; fork 6->9."""
    cfg = copy.deepcopy(load_config())
    cfg["demo_scale"]["main_steps"] = 10
    cfg["checkpoint_every"] = 3
    cfg["crash"]["step"] = 8
    cfg["replay"] = {"from_checkpoint_step": 3, "to_step": 6}
    cfg["fork"].update(from_checkpoint_step=6, steps=3, branch_id="fork-s6-test")
    cfg["firewall_drills"].update(candidate_injection_step=2, direct_batch_injection_step=3)
    return cfg


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def tok(cfg):
    return load_tokenizer(cfg)


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Build phase only (shards, manifests, registries, schedule)."""
    art = tmp_path_factory.mktemp("built")
    r = subprocess.run([sys.executable, "-m", "tdes.build", "--artifacts", str(art)], cwd=REPO, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return art


@pytest.fixture(scope="session")
def demo_small(tmp_path_factory):
    """The full run_demo.py pipeline (every phase in its own process) with the 12-step config."""
    base = tmp_path_factory.mktemp("demo_small")
    cpath = base / "small_config.json"
    cpath.write_text(json.dumps(small_config(), indent=1), encoding="utf-8")
    art = base / "artifacts"
    r = subprocess.run([sys.executable, "run_demo.py", "--config", str(cpath), "--artifacts", str(art)], cwd=REPO,
                       capture_output=True, text=True)
    return {"art": art, "config": cpath, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def run_verify(art, config):
    ev_path = Path(art) / "evidence.json"
    ev_path.unlink()  # never read a stale evidence file
    r = subprocess.run([sys.executable, "-m", "tdes.verify", "--artifacts", str(art), "--config", str(config), "--no-log"],
                       cwd=REPO, capture_output=True, text=True)
    ev = json.loads((Path(art) / "evidence.json").read_text(encoding="utf-8"))
    return r.returncode, {q["id"]: q for q in ev["requirements"]}
