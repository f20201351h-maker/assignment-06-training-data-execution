"""Append-only, hash-chained JSONL ledger + crash recovery.

Each record gets `ledger_offset` (1-based count), `prev_record_hash` and
`record_hash = sha256(canonical(record without record_hash))`. A record is
committed when its full line (with newline) is on disk and fsynced. A
checkpoint stores (ledger_offset, ledger_head_hash), binding model state to a
position in the data history.

Recovery after a crash (truncate to the checkpoint's commit point):
  1. parse the file; a final line without newline / invalid JSON is a torn write;
  2. verify the chain for every complete record;
  3. require record[offset].record_hash == checkpoint head hash (else refuse);
  4. copy the original file byte-for-byte to crash_forensics/ and rewrite the
     canonical file with records 1..offset. Records after the offset were never
     covered by a checkpoint: the model state that consumed them was lost with
     the process, so they are orphaned (kept only in forensics) and those
     steps are trained again from the restored state.
"""
import json
import os
import shutil
from pathlib import Path

from .hashing import canonical_json, sha256_bytes

GENESIS = "0" * 64


class LedgerError(Exception):
    pass


def record_hash(rec) -> str:
    return sha256_bytes(canonical_json({k: v for k, v in rec.items() if k != "record_hash"}))


def _fsync_append(path, data: bytes):
    with open(path, "ab") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


class HashChainLedger:
    def __init__(self, path, genesis_hash=GENESIS, genesis_offset=0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.genesis_hash, self.genesis_offset = genesis_hash, genesis_offset
        recs = self.read(strict=True)["records"] if self.path.exists() else []
        self.head = recs[-1]["record_hash"] if recs else genesis_hash
        self.offset = recs[-1]["ledger_offset"] if recs else genesis_offset

    def encode(self, rec):
        rec = dict(rec)
        rec["ledger_offset"] = self.offset + 1
        rec["prev_record_hash"] = self.head
        rec["record_hash"] = record_hash(rec)
        return rec, (json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")

    def append(self, rec):
        rec, line = self.encode(rec)
        _fsync_append(self.path, line)
        self.head, self.offset = rec["record_hash"], rec["ledger_offset"]
        return rec

    def read(self, strict=False):
        return read_ledger(self.path, self.genesis_hash, self.genesis_offset, strict)


def read_ledger(path, genesis_hash=GENESIS, genesis_offset=0, strict=False):
    raw = Path(path).read_bytes() if Path(path).exists() else b""
    lines = raw.split(b"\n")
    tail = lines.pop()  # bytes after the last newline (b"" if file ends cleanly)
    recs, errors = [], []
    prev, off = genesis_hash, genesis_offset
    for i, ln in enumerate(lines):
        try:
            r = json.loads(ln)
        except Exception:
            errors.append(f"line {i + 1}: invalid JSON")
            break
        if r.get("prev_record_hash") != prev:
            errors.append(f"line {i + 1}: prev_record_hash breaks the chain")
            break
        if record_hash(r) != r.get("record_hash"):
            errors.append(f"line {i + 1}: record_hash does not match content")
            break
        if r.get("ledger_offset") != off + 1:
            errors.append(f"line {i + 1}: offset {r.get('ledger_offset')} != {off + 1}")
            break
        recs.append(r)
        prev, off = r["record_hash"], r["ledger_offset"]
    out = {"records": recs, "torn_tail_bytes": len(tail), "torn_tail_preview": tail[:120].decode("utf-8", "replace"),
           "errors": errors, "head": prev, "offset": off}
    if strict and (errors or tail):
        raise LedgerError(f"{path}: {errors or 'torn tail of %d bytes' % len(tail)}")
    return out


def forensic_path(forensic_dir, name):
    """First crash -> <name>.pre_crash; later crashes get .pre_crash.2, .3 ... so evidence is never overwritten."""
    p = Path(forensic_dir) / (name + ".pre_crash")
    k = 2
    while p.exists():
        p = Path(forensic_dir) / (name + f".pre_crash.{k}")
        k += 1
    return p


def truncate_jsonl_by_step(path, max_step, forensic_dir, step_key="step"):
    """For side ledgers (decisions, learning, perf...): keep lines with step <= max_step."""
    path = Path(path)
    if not path.exists():
        return {"file": path.name, "kept": 0, "dropped": 0, "torn": False}
    raw = path.read_bytes()
    shutil.copyfile(path, forensic_path(forensic_dir, path.name))
    lines = raw.split(b"\n")
    tail = lines.pop()
    keep, dropped = [], 0
    header = None
    if path.suffix == ".csv":
        header, lines = lines[0], lines[1:]
        idx = header.decode().split(",").index(step_key)
    for ln in lines:
        if path.suffix == ".csv":
            s = int(ln.decode("utf-8").split(",")[idx])
        else:
            s = json.loads(ln)[step_key]
        if s <= max_step:
            keep.append(ln)
        else:
            dropped += 1
    body = ([header] if header is not None else []) + keep
    path.write_bytes(b"".join(x + b"\n" for x in body))
    return {"file": path.name, "kept": len(keep), "dropped": dropped, "torn_tail_bytes": len(tail)}


def recover_to_checkpoint(path, ckpt_offset, ckpt_head, forensic_dir):
    path = Path(path)
    forensic_dir = Path(forensic_dir)
    forensic_dir.mkdir(parents=True, exist_ok=True)
    st = read_ledger(path)
    recs = st["records"]
    if st["errors"]:
        raise LedgerError(f"chain broken before recovery: {st['errors']}")
    if ckpt_offset > len(recs):
        raise LedgerError(f"checkpoint offset {ckpt_offset} beyond committed ledger ({len(recs)})")
    head = recs[ckpt_offset - 1]["record_hash"] if ckpt_offset > 0 else GENESIS
    if head != ckpt_head:
        raise LedgerError(f"checkpoint/ledger disagreement at offset {ckpt_offset}: {head} != {ckpt_head}")
    fpath = forensic_path(forensic_dir, path.name)
    shutil.copyfile(path, fpath)
    keep = recs[:ckpt_offset]
    orphan = recs[ckpt_offset:]
    path.write_bytes(b"".join((json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n").encode() for r in keep))
    return {"committed_records_before_recovery": len(recs), "checkpoint_offset": ckpt_offset,
            "checkpoint_head_hash": ckpt_head, "orphaned_records": [r["global_step"] for r in orphan],
            "orphaned_batch_ids": [r["batch_id"] for r in orphan],
            "torn_tail_bytes": st["torn_tail_bytes"], "torn_tail_preview": st["torn_tail_preview"],
            "forensic_copy": f"{forensic_dir.name}/{fpath.name}"}
