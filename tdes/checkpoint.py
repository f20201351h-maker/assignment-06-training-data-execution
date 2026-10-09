"""Checkpoints that bind model state to data state (kept from the earlier prototype; `next_batch` added).

A checkpoint directory holds model.pt, optimizer.pt, state.json (scheduler,
dataloader/planner state, and a hash of the torch RNG state; training uses no
random numbers after initialisation, so the RNG state itself is not needed to resume) and meta.json. meta.json records the step,
ledger offset + head hash, SHA-256 of every file and of the model,
optimizer and dataloader state, and the batch this checkpoint says must come next
(`next_batch`: computed by planning step+1 on a copy of the saved state). It is written last, then the temporary
directory is renamed into place, so a crash during saving never leaves a
directory that looks valid.
"""
import json
import os
import shutil
from pathlib import Path

import torch

from .hashing import sha256_file, sha256_json
from .model import optimizer_hash, state_hash


class CheckpointError(Exception):
    pass


def ckpt_id(branch, step):
    return f"{branch}@s{step:05d}"


def save_checkpoint(root, branch, step, model, opt, sched, planner_state, ledger, extra):
    d = Path(root) / branch / f"step_{step:05d}"
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.save(model.state_dict(), tmp / "model.pt")
    torch.save(opt.state_dict(), tmp / "optimizer.pt")
    state = {"scheduler": sched.state_dict(), "dataloader": planner_state,
             "torch_rng_state_sha256": sha256_json(torch.get_rng_state().tolist())}
    (tmp / "state.json").write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    meta = {"checkpoint_id": ckpt_id(branch, step), "branch": branch, "step": step,
            "ledger_offset": ledger.offset, "ledger_head_hash": ledger.head,
            "model_sha256": state_hash(model.state_dict()), "optimizer_sha256": optimizer_hash(opt),
            "dataloader_state_sha256": sha256_json(planner_state),
            "files": {f: sha256_file(tmp / f) for f in ("model.pt", "optimizer.pt", "state.json")}, **extra}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
    if d.exists():
        raise CheckpointError(f"{d} already exists")
    os.replace(tmp, d)
    return meta


def list_checkpoints(root, branch):
    base = Path(root) / branch
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and not p.name.endswith(".tmp")
                  and (p / "meta.json").exists())


def load_checkpoint(path, model, opt, sched):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    for f, h in meta["files"].items():
        if sha256_file(path / f) != h:
            raise CheckpointError(f"{path / f} hash mismatch")
    model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
    opt.load_state_dict(torch.load(path / "optimizer.pt", weights_only=True))
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    sched.load_state_dict(state["scheduler"])
    if state_hash(model.state_dict()) != meta["model_sha256"]:
        raise CheckpointError("restored model hash differs from meta")
    if optimizer_hash(opt) != meta["optimizer_sha256"]:
        raise CheckpointError("restored optimizer hash differs from meta")
    if sha256_json(state["dataloader"]) != meta["dataloader_state_sha256"]:
        raise CheckpointError("dataloader state hash differs from meta")
    return meta, state["dataloader"]
