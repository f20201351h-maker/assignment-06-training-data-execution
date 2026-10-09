"""Immutable tokenized shards, manifests, the shard registry and the admission gate.

A shard is two flat little-endian arrays (token ids uint16, per-token loss eligibility uint8) plus a document
index. Its identity binds the bytes to the transformation that produced them:

    content_hash = sha256("tdes-shard/2" | tokenizer_id | transform_id | tokens | eligibility | doc_index)
    shard_id     = "shd-<lane>-<content_hash[:12]>"

so the same documents + the same tokenizer + the same ingress config give the same shard id on any run, and a
change to any of them gives a different id. Files are written once (writing different bytes under an existing
id raises ImmutableShardError) and every open recomputes the hash (ShardIntegrityError on mismatch). The
read-only file bit is set too, but it is not what immutability rests on (git does not keep it).
"""
import json
import os
import stat
from pathlib import Path

import numpy as np

from .hashing import canonical_json, sha256_arrays, sha256_bytes, sha256_file, sha256_json

SHARD_FORMAT = "tdes-shard/2"


class ShardIntegrityError(Exception):
    pass


class ImmutableShardError(Exception):
    pass


def content_hash(tokenizer_id, transform_id, tokens, elig, doc_index) -> str:
    return sha256_bytes(SHARD_FORMAT.encode() + tokenizer_id.encode() + transform_id.encode()
                        + bytes.fromhex(sha256_arrays((tokens, "<u2"), (elig, "u1"))) + canonical_json(doc_index))


def _write_readonly(path: Path, data: bytes):
    path.write_bytes(data)
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def store_shard_bytes(sdir: Path, tb: bytes, eb: bytes):
    """Write-once storage. Same bytes again is a no-op; different bytes under an existing id raise."""
    if sdir.exists():
        if (sdir / "tokens.bin").read_bytes() != tb or (sdir / "loss_eligible.bin").read_bytes() != eb:
            raise ImmutableShardError(f"{sdir.name} exists with different bytes; shards are immutable")
        return "identical_existing"
    sdir.mkdir(parents=True)
    _write_readonly(sdir / "tokens.bin", tb)
    _write_readonly(sdir / "loss_eligible.bin", eb)
    return "written"


def write_shard(artifacts: Path, key, docs, tok, common):
    lane, pool, subpool, split = key
    toks, elig, index, off = [], [], [], 0
    for d in docs:
        ids, el = tok.render_document(d)
        index.append({"doc_id": d["doc_id"], "source_id": d["source_id"], "offset": off, "length": len(ids),
                      "n_trainable_tokens": int(sum(el)), "language": d["language"], "script": d.get("script"),
                      "license": d["license"], "a4_record_id": d.get("a4_record_id"), "pii_status": d.get("pii_status"),
                      "doc_content_sha256": sha256_json(d.get("messages") or d["text"])})
        toks.extend(ids)
        elig.extend(el)
        off += len(ids)
    t = np.asarray(toks, dtype=np.uint16)
    e = np.asarray(elig, dtype=np.uint8)
    ch = content_hash(tok.tokenizer_id, common["transform_id"], t, e, index)
    shard_id = f"shd-{lane}-{ch[:12]}"
    tb, eb = t.astype("<u2").tobytes(), e.tobytes()
    store_shard_bytes(artifacts / "shards" / shard_id, tb, eb)
    a4 = sorted({d["source"] for d in docs if d.get("a4_admitted")})
    train = split == "train"
    manifest = {
        "format": SHARD_FORMAT, "shard_id": shard_id, "split": split, "never_train": not train,
        "capability_lane": lane, "lane": lane, "pool": pool, "subpool": subpool,
        "source_ids": sorted({d["source_id"] for d in docs}), "document_ids": [d["doc_id"] for d in docs],
        "doc_index": index, "n_docs": len(docs), "token_count": int(len(t)), "trainable_token_count": int(e.sum()),
        "languages": sorted({d["language"] for d in docs}), "scripts": sorted({d.get("script") or "n/a" for d in docs}),
        "licenses": sorted({d["license"] for d in docs}),
        "provenance": sorted({d.get("source") or d["source_id"] for d in docs}),
        "a5_tier": sorted({d.get("a5_tier") or "n/a" for d in docs}),
        "a4_admitted": bool(a4) and all(d.get("a4_admitted") for d in docs),
        "parent_shard_ids": sorted({common["a4_shard_ids"][s] for s in a4}),
        "cleaning_lineage": (sorted({common["a4_pipeline_ids"][s] for s in a4}) + [common["transform_id"]]),
        "dedup_status": "exact_dedup_passed" if train else "n/a_eval",
        "pii_status": ("n/a" if not train else "redacted" if any(d.get("pii_status") == "redacted" for d in docs) else
                       "exempt_public_source_code" if all(d.get("pii_status") == "exempt_public_source_code" for d in docs)
                       else "clean (exempt public source code inside)" if any(d.get("pii_status") ==
                                                                             "exempt_public_source_code" for d in docs)
                       else "clean"),
        "contamination_status": "clean" if train else "is_eval_data",
        "eval_overlap_status": "none" if train else f"registered_{split}_shard",
        "content_hash": ch,
        "files": {"tokens.bin": {"dtype": "<u2", "sha256": sha256_bytes(tb)},
                  "loss_eligible.bin": {"dtype": "u1", "sha256": sha256_bytes(eb)}},
        "tokenizer_id": tok.tokenizer_id, "a2_tokenizer_sha256": tok.a2_sha256, "transform_id": common["transform_id"],
        "created_by": "tdes.build",
    }
    mpath = artifacts / "manifests" / "shards" / f"{shard_id}.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mbytes = json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8")
    if mpath.exists() and mpath.read_bytes() != mbytes:
        raise ImmutableShardError(f"manifest for {shard_id} exists with different content")
    if not mpath.exists():
        _write_readonly(mpath, mbytes)
    return manifest


def chunk_docs(docs, target_tokens):
    groups, cur, n = [], [], 0
    for d in docs:
        if cur and n + d["n_tokens"] > target_tokens:
            groups.append(cur)
            cur, n = [], 0
        cur.append(d)
        n += d["n_tokens"]
    if cur:
        groups.append(cur)
    return groups


def build_all_shards(artifacts: Path, admitted, eval_docs, tok, common, target_tokens=6000):
    groups = {}
    for d in admitted:
        groups.setdefault((d["lane"], d["pool"], d["subpool"], "train"), []).append(d)
    for d in eval_docs:
        d.setdefault("n_tokens", len(tok.render_document(d)[0]))
        lane = {"test": "benchmark", "validation": "validation", "proxy": "proxy"}[d["split"]]
        groups.setdefault((lane, "eval", "all", d["split"]), []).append(d)
    manifests = []
    for key in sorted(groups):
        docs = sorted(groups[key], key=lambda d: d["doc_id"])
        for chunk in chunk_docs(docs, target_tokens):
            manifests.append(write_shard(artifacts, key, chunk, tok, common))
    return manifests


def admission_gate(m, pinned_tokenizer_id, known_lineages, allowed_licenses, purpose="train"):
    """Reasons this shard may NOT be used for `purpose`; an empty list means admitted."""
    r = []
    if not m.get("tokenizer_id"):
        r.append("missing_tokenizer_hash")
    elif m["tokenizer_id"] != pinned_tokenizer_id:
        r.append("tokenizer_hash_mismatch")
    if not m.get("cleaning_lineage") or any(x not in known_lineages for x in m["cleaning_lineage"]):
        r.append("unknown_cleaning_lineage")
    bad = [x for x in m.get("licenses", []) if x not in allowed_licenses]
    if bad:
        r.append(f"unsafe_license:{bad}")
    if purpose == "train":
        if m.get("split") != "train" or m.get("never_train", True):
            r.append(f"split_not_trainable:{m.get('split')}")
        if m.get("contamination_status") != "clean":
            r.append(f"contamination_status:{m.get('contamination_status')}")
        if m.get("eval_overlap_status") != "none":
            r.append(f"eval_overlap:{m.get('eval_overlap_status')}")
    return r


def write_registry(artifacts: Path, manifests, tokenizer_id):
    entries = []
    for m in sorted(manifests, key=lambda m: m["shard_id"]):
        mp = artifacts / "manifests" / "shards" / f"{m['shard_id']}.json"
        entries.append({"shard_id": m["shard_id"], "split": m["split"], "lane": m["lane"], "pool": m["pool"],
                        "subpool": m["subpool"], "content_hash": m["content_hash"], "manifest_sha256": sha256_file(mp),
                        "token_count": m["token_count"], "n_docs": m["n_docs"],
                        "permissions": {"train": m["split"] == "train", "eval_read": m["split"] == "validation",
                                        "proxy_read": m["split"] == "proxy"}})
    reg = {"format": "tdes-registry/2", "tokenizer_id": tokenizer_id, "shards": entries,
           "registry_sha256": sha256_json(entries)}
    (artifacts / "manifests" / "shard_registry.json").write_text(json.dumps(reg, indent=1, sort_keys=True),
                                                                encoding="utf-8")
    return reg


class ShardStore:
    """Read-only access to shards; verifies registry, manifest, tokenizer id and content hash on open."""

    def __init__(self, artifacts, pinned_tokenizer_id):
        self.artifacts = Path(artifacts)
        self.pinned = pinned_tokenizer_id
        reg = json.loads((self.artifacts / "manifests" / "shard_registry.json").read_text(encoding="utf-8"))
        if reg["tokenizer_id"] != pinned_tokenizer_id:
            raise ShardIntegrityError("registry tokenizer id differs from the pinned tokenizer")
        if sha256_json(reg["shards"]) != reg["registry_sha256"]:
            raise ShardIntegrityError("registry hash mismatch")
        self.registry = {e["shard_id"]: e for e in reg["shards"]}
        self.registry_sha256 = reg["registry_sha256"]
        self.manifests, self._arrays, self._doc = {}, {}, {}

    def manifest(self, shard_id):
        if shard_id not in self.manifests:
            if shard_id not in self.registry:
                raise ShardIntegrityError(f"unknown shard {shard_id}")
            mp = self.artifacts / "manifests" / "shards" / f"{shard_id}.json"
            if sha256_file(mp) != self.registry[shard_id]["manifest_sha256"]:
                raise ShardIntegrityError(f"manifest of {shard_id} changed after registration")
            m = json.loads(mp.read_text(encoding="utf-8"))
            if m["tokenizer_id"] != self.pinned:
                raise ShardIntegrityError(f"{shard_id} was tokenized with {m['tokenizer_id']}, pinned {self.pinned}")
            self.manifests[shard_id] = m
        return self.manifests[shard_id]

    def arrays(self, shard_id):
        if shard_id not in self._arrays:
            m = self.manifest(shard_id)
            sdir = self.artifacts / "shards" / shard_id
            t = np.frombuffer((sdir / "tokens.bin").read_bytes(), dtype="<u2").astype(np.int64)
            e = np.frombuffer((sdir / "loss_eligible.bin").read_bytes(), dtype="u1").astype(np.int64)
            if content_hash(self.pinned, m["transform_id"], t, e, m["doc_index"]) != m["content_hash"] or \
                    m["content_hash"] != self.registry[shard_id]["content_hash"]:
                raise ShardIntegrityError(f"content hash mismatch for {shard_id}")
            self._arrays[shard_id] = (t, e)
            for di in m["doc_index"]:
                self._doc[(shard_id, di["doc_id"])] = di
        return self._arrays[shard_id]

    def doc_info(self, shard_id, doc_id):
        self.arrays(shard_id)
        return self._doc[(shard_id, doc_id)]

    def doc_tokens(self, shard_id, doc_id):
        t, e = self.arrays(shard_id)
        di = self.doc_info(shard_id, doc_id)
        return t[di["offset"]:di["offset"] + di["length"]], e[di["offset"]:di["offset"] + di["length"]]

    def shards_where(self, **kw):
        return sorted(s for s, e in self.registry.items() if all(e.get(k) == v for k, v in kw.items()))
