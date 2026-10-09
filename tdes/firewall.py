"""Evaluation / validation firewall.

Test, validation and proxy shards are registered (content hash, benchmark id, version tag, word n-gram
fingerprints, a canary string, never_train) precisely so the training path can refuse them:

  * ingress gate   - corpus.admit drops any document sharing an n-gram with a registered item;
  * shard gate     - a shard may feed training only if the registry grants `train` and its hash is not a
                     registered eval hash (mixture compiler and candidate generator);
  * content gate   - the decoded text of every packed span is checked against the fingerprints and canary
                     (catches eval text inside a shard that claims to be training data);
  * batch gate     - right before the optimizer step every row goes through both gates again; a violation
                     raises, after the attempt is logged.

The n-gram size is A4's decontamination n (13 words); an item shorter than 13 words is fingerprinted as its
whole word sequence. Validation shards may be read for evaluation only with gradients disabled; every read
is written to eval_access_log.jsonl.
"""
import json
from pathlib import Path

import torch

from .corpus import CANARY, eval_fingerprints, shingle_hashes
from .hashing import sha256_json


class FirewallViolation(Exception):
    pass


def build_eval_registry(artifacts: Path, eval_manifests, eval_docs, ngram):
    by_id = {d["doc_id"]: d for d in eval_docs}
    shards, fps = [], {}
    for m in eval_manifests:
        docs = []
        for di in m["doc_index"]:
            d = by_id[di["doc_id"]]
            k, hs = eval_fingerprints(d["text"], ngram)
            fps.setdefault(str(k), set()).update(hs)
            docs.append({"doc_id": d["doc_id"], "benchmark_id": d.get("benchmark_id"), "item_id": d.get("item_id"),
                         "version": d.get("version"), "fingerprint_n": k, "n_fingerprints": len(hs),
                         "content_sha256": di["doc_content_sha256"]})
        shards.append({"shard_id": m["shard_id"], "split": m["split"], "content_hash": m["content_hash"],
                       "never_train": True, "eval_read_allowed": m["split"] == "validation",
                       "proxy_read_allowed": m["split"] == "proxy",
                       "benchmark_ids": sorted({x["benchmark_id"] for x in docs if x["benchmark_id"]}),
                       "version_tags": sorted({x["version"] for x in docs if x["version"]}), "documents": docs})
    reg = {"format": "tdes-eval-registry/2", "fingerprint": "sha256[:16] of NFC-lowercased word n-grams",
           "ngram_words": ngram, "canaries": [CANARY], "shards": shards,
           "fingerprints": {k: sorted(v) for k, v in sorted(fps.items())}}
    reg["registry_sha256"] = sha256_json(reg)
    (artifacts / "manifests" / "eval_registry.json").write_text(json.dumps(reg, indent=1, sort_keys=True),
                                                                encoding="utf-8")
    return reg


class Firewall:
    def __init__(self, artifacts, shard_store, tokenizer, events_path=None, access_path=None):
        reg = json.loads((Path(artifacts) / "manifests" / "eval_registry.json").read_text(encoding="utf-8"))
        self.fingerprints = {int(k): set(v) for k, v in reg["fingerprints"].items()}
        self.canaries = reg["canaries"]
        self.eval_hashes = {s["content_hash"] for s in reg["shards"]}
        self.eval_shards = {s["shard_id"]: s for s in reg["shards"]}
        self.store, self.tok = shard_store, tokenizer
        self.events_path = Path(events_path) if events_path else None
        self.access_path = Path(access_path) if access_path else None
        self.silent = False  # dry runs (expected-next-batch preview) must not write events

    def _log(self, path, rec):
        if path is not None and not self.silent:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")

    def block(self, gate, step, attempt, reasons, **ctx):
        rec = {"event": "blocked", "gate": gate, "step": step, "attempt": attempt, "reasons": reasons, **ctx}
        self._log(self.events_path, rec)
        return rec

    # ---- gates -----------------------------------------------------------------------------------
    def shard_violations(self, shard_id):
        reg = self.store.registry.get(shard_id)
        if reg is None:
            return ["unregistered_shard"]
        r = []
        if shard_id in self.eval_shards:
            r.append(f"registered_eval_shard:{self.eval_shards[shard_id]['split']}")
        if not reg["permissions"]["train"]:
            r.append(f"no_train_permission:split={reg['split']}")
        if reg["content_hash"] in self.eval_hashes:
            r.append("content_hash_in_eval_registry")
        return r

    def text_violations(self, text):
        r = []
        hits = sum(len(shingle_hashes(text, n) & hs) for n, hs in self.fingerprints.items())
        if hits:
            r.append(f"eval_fingerprint_overlap:{hits}")
        if any(c in text for c in self.canaries):
            r.append("eval_canary")
        return r

    def row_violations(self, row):
        r = []
        for sp in row["spans"]:
            r += [f"{sp['shard_id']}:{x}" for x in self.shard_violations(sp["shard_id"])]
            toks, _ = self.store.doc_tokens(sp["shard_id"], sp["doc_id"])
            text = self.tok.decode(toks[sp["start"]:sp["end"]])
            r += [f"{sp['doc_id']}:{x}" for x in self.text_violations(text)]
        return r

    def check_batch(self, rows, step, branch):
        bad = [{"row": i, "lane": row.get("lane"), "violations": v} for i, row in enumerate(rows)
               if (v := self.row_violations(row))]
        if bad:
            self.block("batch_gate", step, "loss-bearing batch contained registered eval data", bad, branch=branch,
                       action="batch_rejected_before_optimizer_step")
            raise FirewallViolation(f"step {step}: {bad}")
        return True

    # ---- evaluation-only reads ---------------------------------------------------------------------
    def eval_read(self, shard_id, step, branch, purpose, permission="eval_read"):
        reg = self.store.registry[shard_id]
        if not reg["permissions"][permission]:
            raise FirewallViolation(f"{shard_id} has no {permission} permission")
        if torch.is_grad_enabled():
            raise FirewallViolation("validation/proxy data may only be read with gradients disabled")
        self._log(self.access_path, {"event": permission, "shard_id": shard_id, "step": step, "branch": branch,
                                     "purpose": purpose, "grad_enabled": False})
