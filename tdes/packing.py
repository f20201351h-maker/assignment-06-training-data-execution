"""Packed rows: tokens, labels, loss mask, segment ids, position ids, attention.

A row is an ordered list of spans {"shard_id", "doc_id", "start", "end", "pass"} plus its length L; start/end
index the document's rendered tokens (EOS included). The row tensors are a pure function of (spans, L) and
the immutable shards, which is what makes replay from the ledger possible.

Conventions, identical for every policy:
  * segment_ids: 1..k for the k spans in order, 0 for padding;
  * position_ids restart at 0 at the start of every segment;
  * labels[i] = tokens[i+1] when i and i+1 are in the same segment and token i+1 is loss-eligible, else IGNORE;
    loss_mask = labels != IGNORE (no loss across a document boundary, none on padding, none on
    context-only tokens such as user / system / tool-observation turns);
  * attention: position i may attend to j iff j <= i and seg[i] == seg[j] != 0 (padding attends only to itself).

Policies differ in how spans are chosen (dataloader.LaneStream) and in which row length they use:
  concat_chop                    plain text (web, stem, indic): docs + EOS as one stream cut into full rows;
                                 a document may continue in the next row as a new segment
  best_fit_whole_files           code: whole functions only, best fit over a look-ahead window
  whole_trace_next_fit           reasoning: whole traces in stream order, a row closes when the next trace
                                 does not fit (a trace is never cut, so the argument can finish)
  structure_preserving_best_fit  agentic: whole trajectories (turn order intact), loss only on assistant and
                                 tool-call tokens, best fit in the 1024-position long rows
  long_doc_best_fit              long_context: whole documents in the long rows
  pad_only                       validation / proxy: one document per row, right padded
"""
import math

import numpy as np

from .hashing import sha256_arrays, sha256_bytes, sha256_json

IGNORE = -100


def materialize_row(spans, store, L, pad_id):
    tokens = np.full(L, pad_id, dtype=np.int64)
    elig = np.zeros(L, dtype=np.int64)
    seg = np.zeros(L, dtype=np.int64)
    pos = np.zeros(L, dtype=np.int64)
    p = 0
    for k, sp in enumerate(spans, start=1):
        t, e = store.doc_tokens(sp["shard_id"], sp["doc_id"])
        s, en = sp["start"], sp["end"]
        n = en - s
        if not (0 <= s < en <= len(t)) or p + n > L:
            raise ValueError(f"bad span {sp} for a row of length {L} at offset {p}")
        tokens[p:p + n] = t[s:en]
        elig[p:p + n] = e[s:en]
        seg[p:p + n] = k
        pos[p:p + n] = np.arange(n)
        p += n
    same = (seg[:-1] == seg[1:]) & (seg[:-1] > 0)
    loss_mask = np.zeros(L, dtype=np.int64)
    loss_mask[:-1] = same & (elig[1:] == 1)
    labels = np.full(L, IGNORE, dtype=np.int64)
    labels[:-1] = np.where(loss_mask[:-1] == 1, tokens[1:], IGNORE)
    return {"tokens": tokens, "labels": labels, "loss_mask": loss_mask, "segment_ids": seg, "position_ids": pos,
            "L": L, "n_real": int(p), "n_loss": int(loss_mask.sum()),
            "n_context_only": int(p - loss_mask.sum())}  # real positions that carry no loss


def attention_mask(seg):
    seg = np.asarray(seg)
    L = len(seg)
    m = (seg[:, None] == seg[None, :]) & (seg[:, None] > 0) & np.tri(L, dtype=bool)
    m[np.arange(L), np.arange(L)] = True
    return m


def row_hashes(r):
    return {
        "row_hash": sha256_arrays((r["tokens"], "<i4"), (r["labels"], "<i4"), (r["loss_mask"], "u1"),
                                  (r["position_ids"], "<i4"), (r["segment_ids"], "<i4")),
        "tokens_hash": sha256_arrays((r["tokens"], "<i4")),
        "loss_mask_hash": sha256_arrays((r["loss_mask"], "u1")),
        "position_ids_hash": sha256_arrays((r["position_ids"], "<i4")),
        "attention_mask_hash": sha256_bytes(np.packbits(attention_mask(r["segment_ids"])).tobytes()),
    }


def span_id(sp):
    return f"{sp['shard_id']}:{sp['doc_id']}:{sp['start']}-{sp['end']}"


def sample_id(spans, L):
    """Identity of one packed sample: row length + spans, including the pass (a repeat is a different sample)."""
    return "smp-" + sha256_json([L, [[s["shard_id"], s["doc_id"], s["start"], s["end"], s["pass"]] for s in spans]])[:16]


def batch_identity(step, rows):
    """rows: dicts with spans, L, row_hash, loss_mask_hash (in microbatch order)."""
    plan = [[r["L"], [[s["shard_id"], s["doc_id"], s["start"], s["end"]] for s in r["spans"]]] for r in rows]
    return {"batch_id": f"bat-s{step:05d}-" + sha256_json(plan)[:12],
            "batch_hash": sha256_json([r["row_hash"] for r in rows]),
            "batch_loss_mask_hash": sha256_json([r["loss_mask_hash"] for r in rows])}


# ---- what other policies would need for the same documents ------------------------------------------
def rows_needed(lengths, L, policy):
    lengths = [int(x) for x in lengths]
    if not lengths:
        return 0
    if policy == "pad_only":
        return sum(math.ceil(n / L) for n in lengths)
    if policy == "concat_chop":
        return math.ceil(sum(lengths) / L)
    whole = [n for n in lengths if n <= L]
    extra = sum(math.ceil(n / L) for n in lengths if n > L)
    if policy == "next_fit":
        rows, free = 0, 0
        for n in whole:
            if n > free:
                rows, free = rows + 1, L
            free -= n
        return rows + extra
    if policy == "best_fit_decreasing":
        bins = []
        for n in sorted(whole, reverse=True):
            best = None
            for i, free in enumerate(bins):
                if free >= n and (best is None or free < bins[best]):
                    best = i
            if best is None:
                bins.append(L - n)
            else:
                bins[best] -= n
        return len(bins) + extra
    raise ValueError(policy)
