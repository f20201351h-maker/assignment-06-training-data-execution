"""Independent verifier: evidence.json and evidence.md, derived only from files on disk.

    python -m tdes.verify --artifacts artifacts [--no-log]

Nothing here trusts a PASS written by another phase (the one input taken from the orchestrator is the crashed
child's exit code; the torn ledger tail it left behind is checked independently). Each requirement is a list of checks that re-read the
artifacts and recompute the claim with separate, simpler code (its own chain verification, shard hashing,
loop-based row builder, OPUS decision rule, mixture targets from mixture.yaml, eval fingerprints rebuilt from
the raw eval inputs, metrics from raw timings). Where a check could pass vacuously, the same check function
is also run on a deliberately broken copy of real data and must fail ("negative control").
evidence.md is rendered from the same object as evidence.json, so the two cannot disagree.
"""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from .hashing import canonical_json, sha256_arrays, sha256_bytes, sha256_file, sha256_json

REPO = Path(__file__).resolve().parents[1]
IGNORE = -100


# ---------------------------------------------------------------------------------------------------------
class Evidence:
    def __init__(self):
        self.reqs = []

    def req(self, rid, name, table_name, evidence_label, files):
        r = {"id": rid, "name": name, "table_name": table_name, "evidence_label": evidence_label, "files": files,
             "checks": []}
        self.reqs.append(r)
        return r


def check(r, name, ok, detail=""):
    r["checks"].append({"name": name, "pass": bool(ok), "detail": detail})
    return bool(ok)


def safe(r, name, fn):
    try:
        ok, detail = fn()
    except Exception as ex:  # a crash in a check is a failed check, never a pass
        ok, detail = False, f"check raised {type(ex).__name__}: {ex}"
    return check(r, name, ok, detail)


class section:
    """A requirement whose verification raises is recorded as FAIL (with the error), never as a crash or a PASS."""

    def __init__(self, r):
        self.r = r

    def __enter__(self):
        return self.r

    def __exit__(self, et, ev, tb):
        if et is not None:
            check(self.r, "requirement could be verified without an exception", False, f"{et.__name__}: {ev}")
        return True


def jl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def chain(path, genesis_hash="0" * 64, genesis_offset=0):
    """Own hash-chain check: returns (records, errors, torn_tail_bytes)."""
    raw = Path(path).read_bytes()
    lines = raw.split(b"\n")
    tail = lines.pop()
    recs, errs, prev, off = [], [], genesis_hash, genesis_offset
    for i, ln in enumerate(lines):
        r = json.loads(ln)
        body = {k: v for k, v in r.items() if k != "record_hash"}
        if r["prev_record_hash"] != prev or sha256_bytes(canonical_json(body)) != r["record_hash"] \
                or r["ledger_offset"] != off + 1:
            errs.append(f"record {i + 1} breaks the chain")
            break
        recs.append(r)
        prev, off = r["record_hash"], r["ledger_offset"]
    return recs, errs, len(tail)


def rows_of(rec):
    return [row for mb in rec["microbatches"] for row in mb["rows"]]


def sid(sp):
    return f"{sp['shard_id']}:{sp['doc_id']}:{sp['start']}-{sp['end']}"


class Shards:
    """Own reader: registry + manifests + raw arrays, with the content hash recomputed from the bytes."""

    def __init__(self, art):
        self.art = Path(art)
        self.reg = rj(self.art / "manifests" / "shard_registry.json")
        self.entries = {e["shard_id"]: e for e in self.reg["shards"]}
        self.man, self.arr, self.idx = {}, {}, {}

    def manifest(self, s):
        if s not in self.man:
            self.man[s] = rj(self.art / "manifests" / "shards" / f"{s}.json")
        return self.man[s]

    def arrays(self, s):
        if s not in self.arr:
            d = self.art / "shards" / s
            t = np.frombuffer((d / "tokens.bin").read_bytes(), dtype="<u2").astype(np.int64)
            e = np.frombuffer((d / "loss_eligible.bin").read_bytes(), dtype="u1").astype(np.int64)
            self.arr[s] = (t, e)
        return self.arr[s]

    def recompute(self, s):
        m = self.manifest(s)
        t, e = self.arrays(s)
        return sha256_bytes(b"tdes-shard/2" + m["tokenizer_id"].encode() + m["transform_id"].encode()
                            + bytes.fromhex(sha256_arrays((t, "<u2"), (e, "u1"))) + canonical_json(m["doc_index"]))

    def doc(self, s, d):
        if s not in self.idx:
            self.idx[s] = {x["doc_id"]: x for x in self.manifest(s)["doc_index"]}
        di = self.idx[s][d]
        t, e = self.arrays(s)
        return t[di["offset"]:di["offset"] + di["length"]], e[di["offset"]:di["offset"] + di["length"]], di


def ref_row(spans, sh, L, pad):
    """Loop-based reference packer (deliberately not vectorised, shares no code with tdes.packing)."""
    tok, el, seg, pos = [pad] * L, [0] * L, [0] * L, [0] * L
    p = 0
    for k, sp in enumerate(spans, 1):
        t, e, _ = sh.doc(sp["shard_id"], sp["doc_id"])
        for j in range(sp["start"], sp["end"]):
            tok[p], el[p], seg[p], pos[p] = int(t[j]), int(e[j]), k, j - sp["start"]
            p += 1
    lab, lm = [IGNORE] * L, [0] * L
    for i in range(L - 1):
        if seg[i] > 0 and seg[i] == seg[i + 1] and el[i + 1] == 1:
            lab[i], lm[i] = tok[i + 1], 1
    att = np.zeros((L, L), dtype=bool)  # filled block by block: a lower triangle per segment, diagonal for padding
    for k in range(1, len(spans) + 1):
        idx = [i for i in range(L) if seg[i] == k]
        a_, b_ = idx[0], idx[-1] + 1
        att[a_:b_, a_:b_] = np.tril(np.ones((b_ - a_, b_ - a_), dtype=bool))
    for i in range(L):
        if seg[i] == 0:
            att[i, i] = True
    return {"tokens": tok, "labels": lab, "loss_mask": lm, "segment_ids": seg, "position_ids": pos, "attention": att,
            "n_real": p}


def ref_hashes(rr):
    a = lambda k: np.asarray(rr[k])
    return {"row_hash": sha256_arrays((a("tokens"), "<i4"), (a("labels"), "<i4"), (a("loss_mask"), "u1"),
                                      (a("position_ids"), "<i4"), (a("segment_ids"), "<i4")),
            "loss_mask_hash": sha256_arrays((a("loss_mask"), "u1")),
            "position_ids_hash": sha256_arrays((a("position_ids"), "<i4")),
            "attention_mask_hash": sha256_bytes(np.packbits(rr["attention"]).tobytes())}


def words(text):
    out, cur = [], []
    for ch in unicodedata.normalize("NFC", text).lower():
        if unicodedata.category(ch)[0] in "LMN":
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def grams(text, n):
    w = words(text)
    return {hashlib.sha256(" ".join(w[i:i + n]).encode()).hexdigest()[:16] for i in range(len(w) - n + 1)}


# ---------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args(argv)
    art = Path(a.artifacts)
    cfg = json.loads(Path(a.config or REPO / "config" / "demo_config.json").read_text(encoding="utf-8"))
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(REPO / cfg["tokenizer"]["path"]))
    V = tk.get_vocab_size()  # (this call rebuilds the vocabulary each time, so it is cached)
    tman = rj(art / "manifests" / "tokenizer_manifest.json")
    ctrl = tman["spec"]["control_tokens"]
    PAD, EOS = ctrl["<|pad|>"], ctrl["<|eos|>"]
    inv = {v: k for k, v in ctrl.items()}

    def decode(ids):
        out, buf = [], []
        for t in ids:
            if int(t) in inv:
                if buf:
                    out.append(tk.decode(buf))
                    buf = []
                out.append(inv[int(t)])
            else:
                buf.append(int(t))
        if buf:
            out.append(tk.decode(buf))
        return " ".join(out)

    sh = Shards(art)
    plan = rj(art / "manifests" / "a5_plan.json")
    sched = rj(art / "manifests" / "mixture_schedule.json")
    E = Evidence()
    L_main = art / "ledgers" / "main"
    main_recs, main_err, _ = chain(L_main / "consumption_ledger.jsonl")
    ref_dir = art / "reference_run" / "ledgers" / "reference-uninterrupted"
    ref_recs, ref_err, _ = chain(ref_dir / "consumption_ledger.jsonl")
    fkid = cfg["fork"]["branch_id"]
    fdir = art / "ledgers" / fkid
    fpoint = rj(fdir / "fork_point.json")
    fork_recs, fork_err, _ = chain(fdir / "consumption_ledger.jsonl", fpoint["parent_ledger_head_hash"],
                                   fpoint["parent_ledger_offset"])
    all_recs = {"main": main_recs, "reference": ref_recs, "fork": fork_recs}
    decs = jl(L_main / "opus_decisions.jsonl")
    pinned = cfg["tokenizer"]
    facts = {}

    # ===== R01 tokenizer ==================================================================================
    r = E.req("R01", "Frozen tokenizer and content hashes", "Tokenizer integrity", "Manifest record",
              ["manifests/tokenizer_manifest.json", "manifests/shards/*.json", "upstream_contracts.json"])
    with section(r):
        fsha = sha256_file(REPO / pinned["path"])
        a2pub = rj(REPO / "inputs" / "a2" / "final_config.json")["tokenizer_sha256"]
        check(r, "A2 tokenizer.json sha256 == pinned == the hash A2 published", fsha == pinned["a2_sha256"] == a2pub,
              f"{fsha[:16]} / {pinned['a2_sha256'][:16]} / {a2pub[:16]}")
        spec_id = sha256_bytes(canonical_json(tman["spec"]))
        check(r, "adapter tokenizer_id recomputed from its spec == pinned", spec_id == pinned["tokenizer_id"] ==
              tman["tokenizer_id"], spec_id[:16])
        check(r, "spec binds the A2 file hash and the control-token block (ids after the A2 vocabulary)",
              tman["spec"]["a2_tokenizer_sha256"] == fsha and min(ctrl.values()) == V,
              f"control ids {min(ctrl.values())}..{max(ctrl.values())}")
        mans = [sh.manifest(s) for s in sh.entries]
        check(r, "every shard manifest carries the pinned tokenizer_id and A2 hash",
              all(m["tokenizer_id"] == pinned["tokenizer_id"] and m["a2_tokenizer_sha256"] == fsha for m in mans),
              f"{len(mans)} manifests")
        nrec = sum(len(v) for v in all_recs.values())
        check(r, "every consumption-ledger record (main, reference, fork) carries the pinned tokenizer_id",
              nrec > 0 and all(x["tokenizer_id"] == pinned["tokenizer_id"] for v in all_recs.values() for x in v),
              f"{nrec} records")
        probe = tman["determinism_probe"]
        check(r, "deterministic encode: the recorded probe re-encodes to the same ids in this process",
              tk.encode(probe["text"]).ids == probe["ids"], f"{len(probe['ids'])} ids")

        def roundtrip():
            n = bad = 0
            for s, e in sh.entries.items():
                if e["split"] != "train":
                    continue
                for di in sh.manifest(s)["doc_index"]:
                    t, el, _ = sh.doc(s, di["doc_id"])
                    ids = [int(x) for x in t if int(x) < V]
                    if len(ids) != len(t) - 1 or 0 in ids:  # structured docs and docs with <unk> are skipped
                        continue
                    n += 1
                    bad += tk.encode(tk.decode(ids)).ids != ids
            return n > 0 and bad == 0, f"{n} plain-text training documents: encode(decode(ids)) == ids for {n - bad}"
        safe(r, "A2 round trip is stable on stored shard tokens (decode -> encode gives the same ids)", roundtrip)
        mv = rj(art / "manifests" / "manifest_validation.json")
        def reencode_raw():
            """Own loader + own rendering rule: raw vendored text -> ids, compared with the stored shard tokens."""
            inp = REPO / "inputs" / "corpus"
            raw = {f"glaive-{x['a4_id']}": x["messages"] for x in jl(inp / "agentic_glaive.jsonl")}
            raw.update({f"anudesh-{x['a4_id']}": x["text"] for x in jl(inp / "indic_anudesh_prompts.jsonl")})
            raw.update({f"wiki-{x['wiki_id']}": x["text"] for x in jl(inp / "wikipedia_en.jsonl")})
            raw.update({f"py-{x['module']}.{x['function']}": x["text"] for x in jl(inp / "code_stdlib.jsonl")})
            raw.update({f"rs-{x['band_hint']}-{i:03d}": x["text"] for i, x in enumerate(jl(inp / "reasoning_synthetic.jsonl"))})

            def render(src):
                if isinstance(src, str):
                    return tk.encode(src).ids + [EOS]
                out = []
                for m in src:
                    if m.get("tool_call"):
                        out += [ctrl["<|tool_call|>"]] + tk.encode(json.dumps(m["tool_call"], sort_keys=True,
                                                                               ensure_ascii=False)).ids
                    else:
                        out += [ctrl[f"<|{m['role']}|>"]] + tk.encode(m["content"] or "").ids
                return out + [EOS]
            n = skipped = bad = 0
            for s_, e_ in sh.entries.items():
                if e_["split"] != "train":
                    continue
                for di in sh.manifest(s_)["doc_index"]:
                    src = raw.get(di["doc_id"])
                    if src is None or sha256_json(src) != di["doc_content_sha256"]:  # PII-redacted docs differ
                        skipped += 1
                        continue
                    n += 1
                    bad += render(src) != [int(x) for x in sh.doc(s_, di["doc_id"])[0]]
            return n > 0 and bad == 0, (f"{n} training documents re-encoded from inputs/ with the raw A2 tokenizer; "
                                        f"{bad} differ; {skipped} skipped (text changed by PII redaction)")
        safe(r, "every unredacted training document re-encoded independently from the vendored raw text reproduces "
                "its stored shard tokens exactly", reencode_raw)

        def mismatch_probe():
            from .shards import ShardIntegrityError, ShardStore
            variant = sha256_json(dict(tman["spec"], control_tokens=dict(ctrl, **{"<|pad|>": EOS, "<|eos|>": PAD})))
            try:
                ShardStore(art, variant)
                return False, "store accepted a different tokenizer id"
            except ShardIntegrityError as ex:
                return variant != pinned["tokenizer_id"], f"variant {variant[:16]} refused: {ex}"
        safe(r, "a tokenizer that differs only in its control block (EOS/PAD swapped) is refused by the real shard store",
             mismatch_probe)
        facts["tokenizer"] = {"a2_sha256": fsha, "tokenizer_id": pinned["tokenizer_id"], "a2_vocab": V,
                              "control_tokens": ctrl}

    # ===== R10 shards =====================================================================================
    r10 = E.req("R10", "Immutable tokenized shards with manifests", "Shards, manifests & immutability",
                "Recomputed shard hashes", ["shards/", "manifests/shard_registry.json", "manifests/manifest_validation.json"])
    with section(r10):
        bad = [s for s in sh.entries if not (sh.recompute(s) == sh.manifest(s)["content_hash"] == sh.entries[s]["content_hash"]
                                             and s.endswith(sh.manifest(s)["content_hash"][:12]))]
        check(r10, "content hash recomputed from tokens.bin + loss_eligible.bin + tokenizer_id + transform == manifest == "
                   "registry == shard id suffix", not bad and sh.entries, f"{len(sh.entries)} shards, mismatches {bad}")
        check(r10, "registry hash recomputes", sha256_json(sh.reg["shards"]) == sh.reg["registry_sha256"],
              sh.reg["registry_sha256"][:16])
        det = mv["determinism"]
        check(r10, "same inputs + tokenizer + config built twice from scratch give identical shard ids (the second "
                   "build's registry hash equals the registry on disk)",
              det["first_registry_sha256"] == det["second_registry_sha256"] == sh.reg["registry_sha256"]
              and det["same_shard_ids"], f"{det['second_registry_sha256'][:16]} == {sh.reg['registry_sha256'][:16]}")

        def tamper_control():  # own negative control: flip a byte of a copy and recompute
            s0 = next(iter(sh.entries))
            t, e = sh.arrays(s0)
            t2 = t.copy()
            t2[5] ^= 1
            m = sh.manifest(s0)
            h = sha256_bytes(b"tdes-shard/2" + m["tokenizer_id"].encode() + m["transform_id"].encode()
                             + bytes.fromhex(sha256_arrays((t2, "<u2"), (e, "u1"))) + canonical_json(m["doc_index"]))
            import shutil
            import tempfile
            from .shards import ImmutableShardError, ShardIntegrityError, ShardStore, store_shard_bytes
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                shutil.copytree(art / "manifests", td / "manifests")
                shutil.copytree(art / "shards" / s0, td / "shards" / s0)
                tp = td / "shards" / s0 / "tokens.bin"
                tp.chmod(0o666)
                raw_ = bytearray(tp.read_bytes())
                raw_[10] ^= 1
                tp.write_bytes(bytes(raw_))
                try:
                    ShardStore(td, pinned["tokenizer_id"]).arrays(s0)
                    caught = False
                except ShardIntegrityError:
                    caught = True
                eb = (td / "shards" / s0 / "loss_eligible.bin").read_bytes()
                try:
                    store_shard_bytes(td / "shards" / s0, bytes(raw_[:-2]) + bytes(2), eb)
                    refused = False
                except ImmutableShardError:
                    refused = True
            return h != m["content_hash"] and caught and refused, (
                f"own recomputation changes the hash; the real ShardStore rejected the tampered copy: {caught}; "
                f"the real writer refused a rewrite: {refused}")
        safe(r10, "a tampered copy of a shard fails validation under its old identity, and a rewrite is refused "
                  "(own recomputation + the real store and writer on a temporary copy)", tamper_control)
        need = ["shard_id", "source_ids", "document_ids", "tokenizer_id", "token_count", "languages", "scripts",
                "capability_lane", "licenses", "provenance", "cleaning_lineage", "dedup_status", "contamination_status",
                "eval_overlap_status", "content_hash", "parent_shard_ids", "pii_status", "never_train"]
        check(r10, "manifests carry every required provenance field", all(all(k in m for k in need) for m in mans),
              ", ".join(need))
        a4m = [m for m in mans if m.get("a4_admitted")]
        check(r10, "shards built from A4-admitted records name the A4 shard as parent and inherit its cleaning lineage",
              a4m and all(m["parent_shard_ids"] and len(m["cleaning_lineage"]) == 2 for m in a4m),
              f"{len(a4m)} A4-derived shards; parents {sorted({p for m in a4m for p in m['parent_shard_ids']})}")
        adm = rj(art / "manifests" / "admission_report.json")
        dr = adm["dropped_by_reason"]
        check(r10, "the ingress gate actually dropped documents (A2 coverage, exact duplicates, eval contamination)",
              dr.get("a2_tokenizer_coverage", 0) > 0 and dr.get("exact_duplicate_of", 0) > 0 and dr.get("eval_contamination", 0) > 0,
              json.dumps(dr))

        def no_email():
            n = ex = 0
            for s, e in sh.entries.items():
                if e["split"] != "train":
                    continue
                for di in sh.manifest(s)["doc_index"]:
                    if di["pii_status"] == "exempt_public_source_code":
                        ex += 1
                        continue
                    n += 1
                    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", decode(sh.doc(s, di["doc_id"])[0])):
                        return False, f"e-mail address found in {s}/{di['doc_id']}"
            return n > 0, f"{n} training documents decoded and scanned; {ex} public-source-code documents exempt by policy"
        safe(r10, "no e-mail address survives in any PII-screened training document", no_email)

    # ===== R02 firewall ===================================================================================
    r2 = E.req("R02", "Evaluation and validation firewalls", "Evaluation firewall", "Blocked-shard event",
               ["ledgers/build/firewall_events.jsonl", "ledgers/main/firewall_events.jsonl", "manifests/eval_registry.json",
                "manifests/admission_report.json"])
    with section(r2):
        bev = jl(art / "ledgers" / "build" / "firewall_events.jsonl")
        blocked_splits = {sh.entries[e["shard_id"]]["split"] for e in bev if e["gate"] == "mixture_compiler"}
        srcs = {s for l in sched["lanes"].values() for p in l["sources"].values() for v in p.values() for s in v}
        check(r2, "compile time: the misconfigured web source list lost its test AND validation shards",
              blocked_splits >= {"test", "validation"} and all(sh.entries[s]["split"] == "train" for s in srcs),
              f"blocked splits {sorted(blocked_splits)}; {len(srcs)} sources left, all train")
        drill = cfg["firewall_drills"]["ingress_contaminated_document"]
        ing = [d for d in adm["decisions"] if d.get("drill")]
        check(r2, "ingress: a web document carrying a registered test item was dropped for eval contamination",
              ing and ing[0]["decision"] == "dropped" and ing[0]["reasons"][0].startswith("eval_contamination"),
              f"{ing[0]['doc_id']}: {ing[0]['reasons']}" if ing else "no drill record")
        mev = jl(L_main / "firewall_events.jsonl")
        cstep = cfg["firewall_drills"]["candidate_injection_step"]
        cg = [e for e in mev if e["gate"] == "candidate_gate" and e["step"] == cstep]
        fwdec = [d for d in decs if d["reason"] == "eval_firewall"]
        check(r2, f"candidate gate: injected test and validation candidates were blocked before scoring (step {cstep}) "
                  "and recorded as rejected OPUS candidates",
              {e["attempt"].split()[-2] for e in cg} >= {"test", "validation"} and len(fwdec) >= 2,
              f"{len(cg)} events, {len(fwdec)} rejected/eval_firewall decisions")
        bstep = cfg["firewall_drills"]["direct_batch_injection_step"]
        bg = [e for e in mev if e["gate"] == "batch_gate"]
        rec_b = next(x for x in main_recs if x["global_step"] == bstep)
        def batch_gate_live():
            from .firewall import Firewall, FirewallViolation
            from .shards import ShardStore
            from .tokenizer import A2Tokenizer
            st_ = ShardStore(art, pinned["tokenizer_id"])
            fw_ = Firewall(art, st_, A2Tokenizer(REPO / pinned["path"]))  # no event files: nothing is written
            vs = st_.shards_where(split="validation")[0]
            row_ = {"spans": [{"shard_id": vs, "doc_id": st_.manifest(vs)["doc_index"][0]["doc_id"], "start": 0, "end": 32}]}
            good = {"spans": rows_of(main_recs[0])[0]["spans"]}
            try:
                fw_.check_batch([good, row_], 0, "verify")
                return False, "the real batch gate let a validation row through"
            except FirewallViolation:
                return fw_.check_batch([good], 0, "verify"), "re-run now: validation row refused, a real training row accepted"
        safe(r2, "the real batch gate, re-run by the verifier, refuses a validation row and accepts a training row",
             batch_gate_live)
        check(r2, f"batch gate: a validation row placed in the loss-bearing batch was rejected (step {bstep}) and the step "
                  "trained only on train shards",
              bg and bg[0]["step"] == bstep and rec_b["firewall_drill"] == "blocked"
              and all(sh.entries[s]["split"] == "train" for row in rows_of(rec_b) for s in row["shard_ids"]),
              f"event at step {bg[0]['step'] if bg else None}")
        raw_eval = jl(REPO / "inputs" / "eval" / "test_items.jsonl") + jl(REPO / "inputs" / "eval" / "heldout.jsonl")
        n = cfg["cleaning"]["fingerprint_ngram_words"]
        efp = defaultdict(set)
        for e in raw_eval:
            k = min(n, len(words(e["text"])))
            efp[k] |= grams(e["text"], k)

        def scan(recsets):
            spans = hits = 0
            for v in recsets.values():
                for rec in v:
                    for row in rows_of(rec):
                        for sp in row["spans"]:
                            spans += 1
                            if sh.entries[sp["shard_id"]]["split"] != "train" or not sh.entries[sp["shard_id"]]["permissions"]["train"]:
                                return False, f"non-train shard {sp['shard_id']} at step {rec['global_step']}"
                            t, _, _ = sh.doc(sp["shard_id"], sp["doc_id"])
                            txt = decode(t[sp["start"]:sp["end"]])
                            if any(grams(txt, k) & hs for k, hs in efp.items()):
                                hits += 1
            return spans > 0 and hits == 0, f"{spans} consumed spans scanned, {hits} with eval n-grams"
        safe(r2, "no consumed span (main, reference, fork) comes from a non-train shard or shares a word n-gram with any "
                 "raw eval/validation/proxy item (fingerprints rebuilt from inputs/eval, not from the registry)",
             lambda: scan(all_recs))

        def scan_control():
            test_sid = next(s for s, e in sh.entries.items() if e["split"] == "test")
            di = sh.manifest(test_sid)["doc_index"][0]
            bad = copy.deepcopy(main_recs[:1])
            bad[0]["microbatches"][0]["rows"][0]["spans"] = [{"shard_id": test_sid, "doc_id": di["doc_id"], "start": 0,
                                                              "end": min(di["length"], 64), "pass": 1}]
            ok, d = scan({"main": bad})
            return not ok, f"planted test span detected: {d}"
        safe(r2, "negative control: a test span planted into a copy of a ledger row is detected", scan_control)
        item = next(e for e in raw_eval if e.get("benchmark_id") == drill["benchmark_id"] and e.get("item_id") == drill["item_id"])
        item_doc = f"test-{drill['benchmark_id']}-{drill['item_id']}"
        appears = [rec["global_step"] for v in all_recs.values() for rec in v for row in rows_of(rec) for sp in row["spans"]
                   if sp["doc_id"] == item_doc or sp["doc_id"].endswith("-DRILL-contaminated")]
        check(r2, f"the injected eval fixture ({drill['benchmark_id']}/{drill['item_id']}, sha256 "
                  f"{sha256_json(item['text'])[:12]}) never appears in any loss-bearing consumption entry", not appears,
              f"occurrences: {appears}")
        acc = jl(L_main / "eval_access_log.jsonl")
        check(r2, "validation was read for evaluation only, with gradients disabled, and every read is logged",
              acc and all(x["grad_enabled"] is False for x in acc) and all(
                  sh.entries[x["shard_id"]]["permissions"][x["event"]] for x in acc),
              f"{len(acc)} logged reads ({dict(Counter(x['event'] for x in acc))})")
        facts["firewall"] = {"ingress_drill": ing[0] if ing else None, "compile_blocked": [e["shard_id"] for e in bev],
                             "candidate_gate_step": cstep, "batch_gate_step": bstep, "fixture_sha256": sha256_json(item["text"])}

    # ===== R03 packing ====================================================================================
    r3 = E.req("R03", "Packing policies, loss masks, attention masks and position ids", "Packing correctness",
               "Packed-batch report", ["ledgers/main/packing_report.json", "ledgers/main/consumption_ledger.jsonl"])
    with section(r3):
        pol_rows = defaultdict(list)
        for rec in main_recs:
            for row in rows_of(rec):
                pol_rows[row["policy"]].append((rec["global_step"], row))
        examples, mism, sem_bad, checked = {}, [], [], 0

        def semantic(row, rr, policy):
            """Return a list of violated semantic rules for one row."""
            v = []
            L = row["L"]
            segs = rr["segment_ids"]
            for k, sp in enumerate(row["spans"], 1):
                idx = [i for i in range(L) if segs[i] == k]
                if [rr["position_ids"][i] for i in idx] != list(range(len(idx))):
                    v.append("positions do not restart at 0 per segment")
                if idx and rr["loss_mask"][idx[-1]]:
                    v.append("loss across a document boundary")
                t, e, di = sh.doc(sp["shard_id"], sp["doc_id"])
                if policy != "concat_chop" and not (sp["start"] == 0 and sp["end"] == di["length"]):
                    v.append(f"{policy} cut a document ({sp['doc_id']})")
            for i in range(L):
                if segs[i] == 0 and (rr["tokens"][i] != PAD or rr["loss_mask"][i]):
                    v.append("padding is not inert")
                    break
            if policy == "concat_chop" and rr["n_real"] != L:
                v.append("concat_chop row not full")
            if policy == "structure_preserving_best_fit":  # loss only when predicting assistant / tool-call tokens
                role = None
                for i in range(L):
                    t = rr["tokens"][i]
                    if t in inv and inv[t] in ("<|system|>", "<|user|>", "<|assistant|>", "<|tool_call|>", "<|tool|>"):
                        role = inv[t]
                    if i > 0 and rr["loss_mask"][i - 1] and role in ("<|system|>", "<|user|>", "<|tool|>") and t != EOS \
                            and not (t in inv and inv[t] in ("<|assistant|>", "<|tool_call|>")):
                        v.append(f"loss on a {role} token")
                        break
            return v

        for policy, lst in sorted(pol_rows.items()):
            for step, row in lst:
                rr = ref_row(row["spans"], sh, row["L"], PAD)
                hh = ref_hashes(rr)
                checked += 1
                if any(hh[k] != row[k] for k in hh) or sum(rr["loss_mask"]) != row["n_loss_tokens"] or rr["n_real"] != row["n_real_tokens"]:
                    mism.append((step, row["sample_id"]))
                sv = semantic(row, rr, policy)
                if sv:
                    sem_bad.append((step, row["sample_id"], sv[:2]))
                if policy not in examples:
                    L = row["L"]
                    segs = rr["segment_ids"]
                    last = max(i for i in range(L) if segs[i] > 0)
                    examples[policy] = {
                        "step": step, "sample_id": row["sample_id"], "row_len": L, "lane": row["lane"],
                        "segments": [{"segment": k, "doc_id": sp["doc_id"], "span": [sp["start"], sp["end"]],
                                      "loss_positions": sum(1 for i in range(L) if segs[i] == k and rr["loss_mask"][i]),
                                      "context_only_positions": sum(1 for i in range(L) if segs[i] == k and not rr["loss_mask"][i]),
                                      "first_position_ids": [rr["position_ids"][i] for i in range(L) if segs[i] == k][:5]}
                                     for k, sp in enumerate(row["spans"], 1)],
                        "padding_positions": L - rr["n_real"],
                        "last_real_token_attends_to": [int(j) for j in np.nonzero(rr["attention"][last])[0][[0, -1]]],
                        "last_real_token_segment_span": [segs.index(segs[last]), last],
                        "tokens_preview": [(decode([rr["tokens"][i]])[:12], segs[i], rr["position_ids"][i], rr["loss_mask"][i])
                                           for i in range(min(L, 24))]}
        check(r3, "every consumed row rebuilt by an independent loop-based packer from its ledger spans reproduces the "
                  "ledger's row, loss-mask, position and attention hashes", checked > 0 and not mism,
              f"{checked} rows, {len(mism)} mismatches {mism[:3]}")
        check(r3, "semantic rules hold on every row: positions reset per segment, no loss across a boundary, inert padding, "
                  "full concat rows, whole documents for whole-document policies, no loss on user/system/tool tokens",
              checked > 0 and not sem_bad, f"violations: {sem_bad[:3]}")
        check(r3, "attention is causal and block-diagonal: the last real token of each example row attends exactly to its "
                  "own segment", all(e["last_real_token_attends_to"] == e["last_real_token_segment_span"] for e in examples.values()),
              "; ".join(f"{k}: {v['last_real_token_attends_to']}" for k, v in examples.items()))
        agent = [row for _, row in pol_rows.get("structure_preserving_best_fit", [])]
        check(r3, "agentic rows are trained with context-only turns (fewer loss positions than real tokens)",
              agent and all(r_["n_loss_tokens"] < r_["n_real_tokens"] - len(r_["spans"]) for r_ in agent),
              f"{sum(x['n_loss_tokens'] for x in agent)} loss of {sum(x['n_real_tokens'] for x in agent)} real positions")

        def packing_controls():
            step, row = pol_rows["concat_chop"][0]
            out = []
            b = copy.deepcopy(row)
            b["spans"][0]["start"] += 1
            b["spans"][0]["end"] += 1
            out.append(ref_hashes(ref_row(b["spans"], sh, b["L"], PAD))["row_hash"] != row["row_hash"])
            rr = ref_row(row["spans"], sh, row["L"], PAD)
            rr["position_ids"] = list(range(row["L"]))  # no reset at segment starts
            out.append(bool(semantic(row, rr, "concat_chop")) or len(row["spans"]) == 1)
            st2, arow = pol_rows["structure_preserving_best_fit"][0]
            b2 = copy.deepcopy(arow)
            b2["spans"][0]["end"] -= 5  # a cut trajectory
            rr2 = ref_row(b2["spans"], sh, b2["L"], PAD)
            out.append(bool(semantic(b2, rr2, "structure_preserving_best_fit")))
            return all(out), f"shifted span / no position reset / cut trajectory detected: {out}"
        safe(r3, "negative controls: a shifted span, missing position resets and a cut trajectory are all detected",
             packing_controls)
        perf = rj(art / "performance.json")
        pu = {}
        for policy, lst in pol_rows.items():
            for _, row in lst:
                k = f"{policy}@{row['L']}"
                p = pu.setdefault(k, [0, 0, 0, 0])
                p[0] += 1
                p[1] += row["L"]
                p[2] += row["n_real_tokens"]
                p[3] += row["n_loss_tokens"]
        report = {"policies": {k: {"rows": v[0], "positions": v[1], "real": v[2], "loss": v[3], "utilization": v[2] / v[1],
                                   "loss_bearing_fraction": v[3] / v[1],
                                   "pad_only_utilization_same_docs": perf["branches"]["main"]["by_policy"][k][
                                       "counterfactual_utilization_for_same_documents"]["pad_only"]}
                               for k, v in sorted(pu.items())},
                  "examples": examples, "rows_checked": checked}
        (L_main / "packing_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True),
                                                    encoding="utf-8")
        check(r3, "per-policy utilization recomputed from the ledger equals performance.json",
              all(abs(perf["branches"]["main"]["by_policy"][k]["utilization"] - v["utilization"]) < 1e-12
                  for k, v in report["policies"].items()), f"{len(report['policies'])} policy/row-length groups")
        facts["packing"] = report["policies"]

    # ===== R04 mixture ====================================================================================
    r4 = E.req("R04", "Curriculum stages, lane weights and protected floors", "Mixture compliance",
               "Planned versus actual shares", ["manifests/mixture_schedule.json", "ledgers/main/mixture_report.json",
                                                "inputs/a5/mixture.yaml"])
    with section(r4):
        a5 = yaml.safe_load((REPO / cfg["mixture_contract"]).read_text(encoding="utf-8"))
        ds = cfg["demo_scale"]
        names = list(a5["stages"])
        spans_ = {s: a5["stages"][s]["span"] * ds["main_steps"] for s in names}
        fl = {s: math.floor(v) for s, v in spans_.items()}
        for s in sorted(names, key=lambda s: -(spans_[s] - fl[s]))[:ds["main_steps"] - sum(fl.values())]:
            fl[s] += 1
        bounds, st0 = {}, 1
        for s in names:
            bounds[s] = (st0, st0 + fl[s] - 1)
            st0 += fl[s]
        bounds["anneal"] = (st0, st0 + max(ds["anneal_steps_min"], round(ds["main_steps"] * a5["budget"]["anneal"] /
                                                                          a5["budget"]["main_run"])) - 1)
        check(r4, "stage boundaries recomputed from A5 spans match the compiled plan",
              all(bounds[s["name"]] == (s["step_start"], s["step_end"]) for s in plan["stages"]),
              "; ".join(f"{k} {v[0]}-{v[1]}" for k, v in bounds.items()))
        mixes = {s: a5["stages"][s]["mix"] for s in names}
        mixes["anneal"] = a5["anneal_mix"]
        order = names + ["anneal"]
        w = max(ds["ramp_steps_min"], round(a5["transition_ramp_tokens"] / a5["budget"]["main_run"] * ds["main_steps"]))

        def mix_at(step):
            k = next(i for i, s in enumerate(order) if bounds[s][0] <= step <= bounds[s][1])
            for j in range(1, len(order)):
                lo = bounds[order[j]][0] - w // 2
                if lo <= step <= lo + w - 1:
                    al = (step - lo + 1) / (w + 1)
                    return order[k], {l: (1 - al) * mixes[order[j - 1]][l] + al * mixes[order[j]][l] for l in a5["lanes"]}
            return order[k], dict(mixes[order[k]])
        bad_mix = [s["step"] for s in sched["steps"] if any(abs(mix_at(s["step"])[1][l] - s["mix_pct"][l]) > 1e-6 for l in a5["lanes"])]
        check(r4, "per-step effective mixture (stage mix, linear ramps centred on boundaries) recomputed from mixture.yaml "
                  "equals the compiled schedule", not bad_mix, f"ramp width {w} steps; mismatching steps {bad_mix}")
        by_step = {x["global_step"]: x for x in main_recs}
        got_q = [s["step"] for s in sched["steps"] if
                 {l: n for l, n in s["quotas"].items() if n} != {l: n for l, n in Counter(r_["lane"] for r_ in rows_of(by_step[s["step"]])).items()}]
        check(r4, "rows actually consumed per lane at every step == the compiled quotas (OPUS never moves a slot across lanes)",
              not got_q and len(by_step) == len(sched["steps"]), f"{len(sched['steps'])} steps; mismatches {got_q}")
        prot = a5["protected"]
        viol, cum, seg = [], {}, None
        for s in sched["steps"]:
            stage, _ = mix_at(s["step"])
            if stage != seg:
                seg, cum = stage, {"total": 0, **{l: 0 for l in a5["lanes"]}}
            rec = by_step[s["step"]]
            for row in rows_of(rec):
                cum[row["lane"]] += row["L"]
                cum["total"] += row["L"]
            for l in prot:
                if cum[l] < mixes[stage][l] / 100 * cum["total"] - 1e-9:
                    viol.append((s["step"], l, cum[l], mixes[stage][l] / 100 * cum["total"]))
        check(r4, "protected floors (indic, reasoning, agentic = their A5 stage share): the stage-cumulative share of "
                  "consumed token positions is at or above the floor after every step", not viol, f"violations {viol[:3]}")
        ov = [d for d in decs if d["protected_floor_override"]]
        trained_ids = {row["opus_decision_id"] for rec in main_recs for row in rows_of(rec)}
        check(r4, "the floor override actually executed: protected rows OPUS would have rejected were trained",
              ov and all(d["decision_id"] in trained_ids for d in ov),
              f"{len(ov)} overrides, lanes {dict(Counter(d['lane'] for d in ov))}")
        shares = {}
        for stage in order:
            tgt, act = defaultdict(float), defaultdict(int)
            for s in sched["steps"]:
                if s["stage"] != stage:
                    continue
                for l in a5["lanes"]:
                    tgt[l] += s["target_positions"][l]
                for row in rows_of(by_step[s["step"]]):
                    act[row["lane"]] += row["L"]
            T, A = sum(tgt.values()), sum(act.values())
            shares[stage] = {l: {"a5_stage_pct": mixes[stage][l], "planned_pct_incl_ramps": round(100 * tgt[l] / T, 3),
                                 "actual_pct": round(100 * act[l] / A, 3), "actual_positions": act[l]} for l in a5["lanes"]}
        run_t, run_a = defaultdict(float), defaultdict(int)
        for s in sched["steps"]:
            for l in a5["lanes"]:
                run_t[l] += s["target_positions"][l]
            for row in rows_of(by_step[s["step"]]):
                run_a[row["lane"]] += row["L"]
        P = cfg["demo_scale"]["regular_positions_per_step"]
        LONG = [l for l in a5["lanes"] if cfg["lanes"][l]["row"] == "long"]

        def rounding_bounds(recs_by_step):
            """Step-wise, cumulative within each stage, in token positions, against targets recomputed here from
            mixture.yaml: long_context within half a long row of its target; a protected lane at most one row above
            max(target, floor); an unprotected regular lane at most one row above its target and below it by at most
            what the protected lanes took above theirs plus one row."""
            bad, seg, worst = [], None, {}
            for s_ in sched["steps"]:
                stage, mx = mix_at(s_["step"])
                if stage != seg:
                    seg, tg, ac, tot, Lmax = stage, defaultdict(float), defaultdict(int), 0, 0
                mlong = sum(mx[l] for l in LONG) / 100
                for l in a5["lanes"]:
                    tg[l] += P * mx[l] / 100 / (1 - mlong)
                for row in rows_of(recs_by_step[s_["step"]]):
                    ac[row["lane"]] += row["L"]
                    tot += row["L"]
                Lmax = max(Lmax, s_["base_window"])
                excess = sum(max(0.0, ac[l] - tg[l]) for l in prot if cfg["lanes"][l]["row"] == "base")
                for l in a5["lanes"]:
                    d = ac[l] - tg[l]
                    worst[l] = max(worst.get(l, 0.0), abs(d))
                    if l in prot:
                        rl = 1024 if l in LONG else Lmax
                        ok = ac[l] <= max(tg[l], mixes[stage][l] / 100 * tot) + rl + 1e-6
                    elif l in LONG:
                        ok = abs(d) <= 512 + 1e-6
                    else:
                        ok = d <= Lmax + 1e-6 and -d <= excess + Lmax + 1e-6
                    if not ok:
                        bad.append((s_["step"], l, round(d, 1)))
            return not bad, (f"every step checked; largest |actual - target| in positions per lane "
                             f"{ {k: round(v) for k, v in worst.items()} }; violations {bad[:4]}")
        safe(r4, "rounding is bounded at every step: each lane's consumed positions stay within whole-row rounding of "
                 "its A5 target (recomputed here from mixture.yaml); stage-level deviations are listed in "
                 "mixture_report.json", lambda: rounding_bounds(by_step))

        def mix_control():
            b = copy.deepcopy(by_step)
            s_ = next(st["step"] for st in sched["steps"] if st["stage"] == "S2_broaden" and st["ramp"] is None)
            for r_ in [r_ for r_ in rows_of(b[s_]) if r_["lane"] == "web"][:2]:
                r_["lane"] = "code"  # two rows silently moved from web to code in one S2 step
            lc = next(st["step"] for st in sched["steps"] if st["quotas"]["long_context"])
            c = copy.deepcopy(by_step)
            for r_ in rows_of(c[lc]):
                if r_["lane"] == "long_context":
                    r_["lane"] = "web"  # one long-context row lost to web
            return (not rounding_bounds(b)[0]) and (not rounding_bounds(c)[0]), \
                "two web rows relabelled as code, and one long-context row relabelled as web, are both caught"
        safe(r4, "negative control: a two-row lane drift and a lost long-context row are not explained by rounding",
             mix_control)
        win = [s["step"] for s in sched["steps"] if s["base_window"] != next(x for x in plan["stages"] if x["name"] == s["stage"])["seq_len"]]
        check(r4, "the window moves before the mixture ramp (A5: never at the same step): base window 256 starts before S3",
              win and max(win) < bounds["S3_reasoning_code"][0], f"steps with the next stage's window: {win}")
        early = []
        for rec in main_recs:
            for row in rows_of(rec):
                if row["pool"] == "reserve" and rec["stage"] != "anneal":
                    early.append(rec["global_step"])
                if rec["stage"] == "anneal" and row["lane"] in plan["reserve_fraction"] and row["pool"] != "reserve":
                    early.append(("anneal_not_from_reserve", rec["global_step"], row["lane"]))
        check(r4, "anneal reserves (A5 section 7) are first seen in the anneal, and the anneal draws those lanes from them",
              not early, f"{early[:3]}")
        bands = defaultdict(Counter)
        for rec in main_recs:
            for row in rows_of(rec):
                if row["lane"] in ("indic", "reasoning"):
                    bands[f"{row['lane']}/{rec['stage']}"][row["subpool"]] += 1
        check(r4, "sub-pools execute: Indic difficulty bands shift harder over stages and reasoning uses only A5-gated bands "
                  "(substitutions listed in the schedule)",
              all(b in ("low", "medium") for k, c in bands.items() if k.startswith("reasoning") for b in c)
              and bands["indic/S1_foundation"]["B0_B1"] >= bands["indic/S1_foundation"].get("B4_B5", 0),
              json.dumps({k: dict(v) for k, v in sorted(bands.items())}))
        mrep = {"stage_shares": shares, "run_planned_pct": {l: 100 * v / sum(run_t.values()) for l, v in run_t.items()},
                "run_actual_pct": {l: 100 * v / sum(run_a.values()) for l, v in run_a.items()},
                "subpools": {k: dict(v) for k, v in bands.items()}, "band_substitutions": sched["band_substitutions"],
                "scaling_notes": plan["scaling_notes"], "floor_rounding": "protected lanes round up against consumed positions"}
        (L_main / "mixture_report.json").write_text(json.dumps(mrep, indent=1, sort_keys=True), encoding="utf-8")
        facts["mixture"] = mrep

    # ===== R05 OPUS =======================================================================================
    r5 = E.req("R05", "OPUS acceptance, rejection, deferral and protected-floor override", "OPUS audit trail",
               "Candidate decision records", ["ledgers/main/opus_decisions.jsonl"])
    with section(r5):
        cnt = Counter(d["status"] for d in decs)
        cnt["protected_floor_override"] = len(ov)
        check(r5, "all four outcomes occur: accepted, rejected, deferred, protected-floor override",
              all(cnt[k] > 0 for k in ("accepted", "rejected", "deferred", "protected_floor_override")), json.dumps(cnt))

        def rederive(ds_):
            bad = []
            bystep = defaultdict(list)
            for d in ds_:
                bystep[(d["step"], d["branch"])].append(d)
            for (step, _), dd in bystep.items():
                stg = dd[0]["stage"]
                sc = [d for d in dd if d["opus_score"] is not None]
                web = sorted([d for d in sc if d["lane"] == "web"], key=lambda d: (-d["opus_score"], dd.index(d)))
                cut = None
                if stg != "anneal" and web:
                    q = web[0]["lane_quota"]
                    band = math.ceil(cfg["opus"]["defer_band_factor"] * q)
                    for i, d in enumerate(web):
                        exp = (("accepted", "opus_selected") if i < q else
                               (("rejected", "defer_expired") if d["deferrals_before"] >= cfg["opus"]["max_deferrals"]
                                else ("deferred", "marginal_utility")) if i < q + band else ("rejected", "low_proxy_utility"))
                        got = (d["status"], d["reason"])
                        if got != exp and not (exp[0] == "deferred" and got == ("rejected", "defer_queue_full")):
                            bad.append((step, d["decision_id"], got, exp))
                    cut = web[q - 1]["opus_score"]
                for d in sc:
                    if d["lane"] == "web" and stg != "anneal":
                        continue
                    if stg == "anneal":
                        exp = "selector_off_anneal"
                    elif d["lane"] in prot:
                        exp = "protected_floor_override" if cut is not None and d["opus_score"] < cut else "always_on_above_cutoff"
                    elif plan["selector_keep"].get(d["lane"]) == 1.0:
                        exp = "keep_1.0_static_filter_only"
                    else:
                        exp = "selector_bypass_long_context"
                    if (d["status"], d["reason"]) != ("accepted", exp):
                        bad.append((step, d["decision_id"], d["reason"], exp))
            return bad
        bad5 = rederive(decs)
        check(r5, "every decision re-derived from the recorded scores with the documented A5 rule (own implementation)",
              not bad5 and len(decs) > 0, f"{len(decs)} decisions, {len(bad5)} disagree {bad5[:2]}")

        def opus_control():
            d2 = copy.deepcopy(decs)
            i = next(i for i, d in enumerate(d2) if d["reason"] == "opus_selected")
            d2[i]["status"], d2[i]["reason"] = "rejected", "low_proxy_utility"
            j = next(j for j, d in enumerate(d2) if d["reason"] == "protected_floor_override")
            d2[j]["opus_score"] = 1.0
            return len(rederive(d2)) >= 2, "a flipped selection and a fake override are both caught"
        safe(r5, "negative control: a flipped selection and a forged override fail re-derivation", opus_control)
        acc_ids = {d["decision_id"]: d for d in decs if d["status"] == "accepted"}
        led_ids = [row["opus_decision_id"] for rec in main_recs for row in rows_of(rec)]
        att = {rec["global_step"]: rec["attempt"] for rec in main_recs}
        accepted_final = [d for d in acc_ids.values()]
        check(r5, "every trained row points to exactly one accepted decision with the same candidate, and every accepted "
                  "decision is trained exactly once", len(led_ids) == len(set(led_ids)) and set(led_ids) == set(acc_ids)
              and all(acc_ids[row["opus_decision_id"]]["candidate_id"] == row["candidate_id"] for rec in main_recs for row in rows_of(rec)),
              f"{len(led_ids)} rows, {len(accepted_final)} accepted decisions")
        model_before = {rec["global_step"]: rec["model_sha256_before"] for rec in main_recs}
        check(r5, "each decision names the model that scored it (= the model before that step in the ledger) and the proxy",
              all(d["scoring_model"]["model_sha256"] == model_before[d["step"]] for d in decs) and
              len({d["proxy_version"]["proxy_batch_sha256"] for d in decs}) == 1, "")
        fate = defaultdict(list)
        for d in decs:
            fate[d["candidate_id"]].append(d)
        final_q = {}
        lastck = sorted((art / "checkpoints" / "main").iterdir())[-1]
        for l, q in rj(lastck / "state.json")["dataloader"]["deferred"].items():
            for e in q:
                final_q[sha256_json([e["L"], [[s["shard_id"], s["doc_id"], s["start"], s["end"], s["pass"]] for s in e["spans"]]])[:16]] = 1
        lost = [c for c, ds_ in fate.items() if ds_[-1]["status"] == "deferred" and c[5:] not in final_q]
        check(r5, "deferred data does not disappear: every deferral is re-scored later, ends rejected with a reason, or is "
                  "still queued in the final checkpoint", not lost,
              f"{sum(1 for v in fate.values() if any(x['status'] == 'deferred' for x in v))} deferred candidates; "
              f"{sum(1 for v in fate.values() if len(v) > 1)} re-scored; lost {lost[:3]}")
        facts["opus"] = {"counts": dict(cnt), "by_reason": dict(Counter(f"{d['status']}:{d['reason']}" for d in decs)),
                         "overrides_by_lane": dict(Counter(d["lane"] for d in ov))}

    # ===== R06 crash recovery =============================================================================
    r6 = E.req("R06", "Crash recovery without skipped or repeated batches", "Crash recovery",
               "Expected and resumed batch ids", ["ledgers/crash_report.json", "ledgers/main/recovery_report.json",
                                                  "ledgers/main/crash_forensics/"])
    with section(r6):
        cr = rj(art / "ledgers" / "crash_report.json")
        rr_ = rj(L_main / "recovery_report.json")
        check(r6, "the training process really died (exit code) while writing the crash step's ledger record",
              cr["child_exit_code"] == cr["expected_exit_code"] == cfg["crash"]["exit_code"] and cr["torn_tail_bytes"] > 0,
              f"exit {cr['child_exit_code']}, torn tail {cr['torn_tail_bytes']} bytes")
        fpre = sorted((L_main / "crash_forensics").glob("consumption_ledger.jsonl.pre_crash*"))[0]
        pre_recs, pre_err, pre_tail = chain(fpre)
        ck_step = int(rr_["checkpoint_step"])
        orphan = [x["global_step"] for x in pre_recs if x["global_step"] > ck_step]
        check(r6, "the forensic copy keeps the torn tail and the steps no checkpoint covered",
              pre_tail == cr["torn_tail_bytes"] and orphan == rr_["orphaned_records"] == cr["steps_committed_after_latest_checkpoint"],
              f"orphaned steps {orphan}")
        ckm = rj(art / "checkpoints" / "main" / f"step_{ck_step:05d}" / "meta.json")
        nxt = ckm["next_batch"]
        res_rec = by_step[ck_step + 1]
        got = {"batch_id": res_rec["batch_id"], "batch_hash": res_rec["batch_hash"],
               "sample_ids": [x["sample_id"] for x in rows_of(res_rec)],
               "span_ids": [[sid(s) for s in x["spans"]] for x in rows_of(res_rec)],
               "row_hashes": [x["row_hash"] for x in rows_of(res_rec)]}
        pre_n = next(x for x in pre_recs if x["global_step"] == ck_step + 1)
        exp_pre = {"batch_id": pre_n["batch_id"], "batch_hash": pre_n["batch_hash"],
                   "sample_ids": [x["sample_id"] for x in rows_of(pre_n)],
                   "span_ids": [[sid(s) for s in x["spans"]] for x in rows_of(pre_n)],
                   "row_hashes": [x["row_hash"] for x in rows_of(pre_n)]}
        keys = ("batch_id", "batch_hash", "sample_ids", "span_ids", "row_hashes")
        m1 = all(nxt[k] == got[k] for k in keys)
        m2 = all(exp_pre[k] == got[k] for k in keys)
        check(r6, f"first resumed batch (step {ck_step + 1}) == the next batch declared in checkpoint {ckm['checkpoint_id']} "
                  "(batch id, hash, sample ids, spans, row hashes)", m1 and res_rec["attempt"] == 2,
              f"declared {nxt['batch_id']} / {nxt['batch_hash'][:16]}; resumed {got['batch_id']} / {got['batch_hash'][:16]}")
        check(r6, "first resumed batch == the batch the dead process had already committed for that step", m2,
              f"pre-crash {exp_pre['batch_id']}")
        steps = [x["global_step"] for x in main_recs]

        def no_skip_dup(main_l, ref_l):
            s_ = [x["global_step"] for x in main_l]
            if s_ != list(range(1, len(ref_l) + 1)):
                return False, f"steps {s_[:5]}... not exactly 1..{len(ref_l)}"
            diff = [x["global_step"] for x, y in zip(main_l, ref_l) if (x["batch_id"], x["batch_hash"], x["model_sha256_after"])
                    != (y["batch_id"], y["batch_hash"], y["model_sha256_after"])]
            ms = Counter(x["sample_id"] for rec in main_l for x in rows_of(rec))
            rs = Counter(x["sample_id"] for rec in ref_l for x in rows_of(rec))
            return not diff and ms == rs, f"{len(s_)} steps; differing from the uninterrupted run: {diff}; sample multisets equal: {ms == rs}"
        safe(r6, "no skip, no repeat: the recovered ledger has every step exactly once, and batch ids, hashes and the model "
                 "after every step equal the uninterrupted reference run", lambda: no_skip_dup(main_recs, ref_recs))

        def crash_controls():
            sk = [x for x in main_recs if x["global_step"] != ck_step + 1]
            dp = main_recs[:ck_step + 1] + [main_recs[ck_step]] + main_recs[ck_step + 1:]
            return (not no_skip_dup(sk, ref_recs)[0]) and (not no_skip_dup(dp, ref_recs)[0]), "a skipped and a repeated batch are both caught"
        safe(r6, "negative control: a skipped batch and a repeated batch both fail", crash_controls)
        check(r6, "the recovery verified checkpoint <-> ledger agreement before trimming (head hash at the offset)",
              pre_recs[ck_step - 1]["record_hash"] == ckm["ledger_head_hash"] == rr_["checkpoint_head_hash"], ckm["ledger_head_hash"][:16])
        facts["crash"] = {"crash_step": cfg["crash"]["step"], "exit_code": cr["child_exit_code"], "torn_tail_bytes": cr["torn_tail_bytes"],
                          "checkpoint": ckm["checkpoint_id"], "ledger_offset": ckm["ledger_offset"], "orphaned": orphan,
                          "expected_next_batch": nxt["batch_id"], "expected_hash": nxt["batch_hash"],
                          "resumed_batch": got["batch_id"], "resumed_hash": got["batch_hash"], "samples": got["sample_ids"],
                          "resume_latency_s": rr_["resume_latency_s"]}

    # ===== R07 replay =====================================================================================
    s0, s1 = cfg["replay"]["from_checkpoint_step"], cfg["replay"]["to_step"]
    rdir = art / "ledgers" / f"replay-main-s{s0}-s{s1}"
    r7 = E.req("R07", "Replay of the same historical data stream", "Replay", "Original and replay hashes",
               [f"ledgers/replay-main-s{s0}-s{s1}/consumption_ledger.jsonl", f"ledgers/replay-main-s{s0}-s{s1}/replay_report.json"])
    with section(r7):
        rp_recs, rp_err, _ = chain(rdir / "consumption_ledger.jsonl")
        toks = defaultdict(list)
        with open(L_main / "learning_tokens.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                st = int(row["step"])
                if s0 < st <= s1:
                    toks[st].append([int(row["row"]), int(row["target_pos"]), row["loss"]])

        def replay_eq(rp):
            if [x["global_step"] for x in rp] != list(range(s0 + 1, s1 + 1)):
                return False, "replayed steps do not cover the interval"
            bad = []
            for x in rp:
                o = by_step[x["global_step"]]
                if not (x["batch_id"] == o["batch_id"] and x["batch_hash"] == o["batch_hash"]
                        and x["batch_loss_mask_hash"] == o["batch_loss_mask_hash"]
                        and x["span_ids"] == [[sid(s) for s in row["spans"]] for row in rows_of(o)]
                        and x["row_hashes"] == [row["row_hash"] for row in rows_of(o)]
                        and x["token_loss_digest"] == sha256_json(sorted(toks[x["global_step"]]))
                        and x["step_loss"] == o["step_loss"] and x["model_sha256_after"] == o["model_sha256_after"]):
                    bad.append(x["global_step"])
            return not bad, f"{len(rp)} steps compared field by field with the main ledger; mismatches {bad}"
        check(r7, "replay ledger hash chain verifies", not rp_err and rp_recs, f"{len(rp_recs)} records")
        safe(r7, f"replayed steps {s0 + 1}-{s1}: batch ids, token spans, row hashes, batch and loss-mask hashes, per-token "
                 "loss digest (recomputed from the main CSV), step loss and model hash equal the original", lambda: replay_eq(rp_recs))

        def fp(pairs):
            h = "0" * 64
            for b, hh in pairs:
                h = sha256_json([h, b, hh])
            return h
        fo = fp([(by_step[s]["batch_id"], by_step[s]["batch_hash"]) for s in range(s0 + 1, s1 + 1)])
        fr = fp([(x["batch_id"], x["batch_hash"]) for x in rp_recs])
        check(r7, "stream fingerprint (rolling sha256 over ordered batch id + hash) of the interval: original == replay",
              fo == fr, f"{fr[:16]} == {fo[:16]}")
        rpr = rj(rdir / "replay_report.json")
        endm = rj(art / "checkpoints" / "main" / f"step_{s1:05d}" / "meta.json")
        check(r7, f"after the replay the model and optimizer equal checkpoint main@s{s1:05d}",
              rpr["final_state"]["replayed_model_sha256"] == endm["model_sha256"] and
              rpr["final_state"]["replayed_optimizer_sha256"] == endm["optimizer_sha256"], endm["model_sha256"][:16])

        def replay_control():
            b = copy.deepcopy(rp_recs)
            b[2]["span_ids"][0][0] = b[2]["span_ids"][0][0].replace("-", "+", 1)
            c = copy.deepcopy(rp_recs)[:-1]
            return (not replay_eq(b)[0]) and (not replay_eq(c)[0]), "an altered span and a missing step are both caught"
        safe(r7, "negative control: an altered span and a missing replay step both fail", replay_control)
        facts["replay"] = {"interval": [s0 + 1, s1], "from": rpr["restored_checkpoint"], "fingerprint": fr,
                           "batch_ids": [x["batch_id"] for x in rp_recs], "cross_check_replan": rpr["cross_check_all_match"],
                           "replay_latency_s": rpr["replay_latency_s"]}

    # ===== R08 learning trace =============================================================================
    r8 = E.req("R08", "Training consumption and learning ledgers, token-level loss", "Learning trace",
               "Loss linked to source data", ["ledgers/main/learning_tokens.csv", "ledgers/main/learning_samples.jsonl",
                                              "ledgers/learning_shards.json", "ledgers/main/eval_checkpoints.jsonl"])
    with section(r8):
        raw_lines = defaultdict(list)
        with open(L_main / "learning_tokens.csv", encoding="utf-8", newline="") as f:
            header = f.readline()
            rd = csv.reader(f)
            rows_csv = []
            for row in rd:
                rows_csv.append(row)
        cols = header.strip().split(",")
        ix = {c: i for i, c in enumerate(cols)}
        for row in rows_csv:
            raw_lines[int(row[ix["step"]])].append(row)

        def trace_hash(lines):
            import io
            buf = io.StringIO()
            w_ = csv.writer(buf, lineterminator="\n")
            for x in lines:
                w_.writerow(x)
            return sha256_bytes(buf.getvalue().encode("utf-8"))
        th_bad = [rec["global_step"] for rec in main_recs if trace_hash(raw_lines[rec["global_step"]]) != rec["learning_trace_sha256"]]
        check(r8, "each consumption record's learning_trace_sha256 equals the hash of that step's token lines (the two "
                  "ledgers are bound)", not th_bad, f"{len(main_recs)} steps, mismatches {th_bad}")
        cnt_bad = [rec["global_step"] for rec in main_recs if len(raw_lines[rec["global_step"]]) != rec["n_loss_tokens"]]
        check(r8, "one token line per loss-bearing position of every step (no more, no less)", not cnt_bad,
              f"{len(rows_csv)} token lines; mismatching steps {cnt_bad}")

        def lookup():
            bad = n = 0
            for row in rows_csv:
                t, _, di = sh.doc(row[ix["shard_id"]], row[ix["doc_id"]])
                n += 1
                lo = float(row[ix["loss"]])
                if int(t[int(row[ix["doc_offset"]])]) != int(row[ix["token_id"]]) or not math.isfinite(lo) or lo < 0 \
                        or abs(math.exp(lo) - float(row[ix["ppl"]])) > 1e-4 * math.exp(lo):
                    bad += 1
            return n > 0 and bad == 0, f"{n} token lines looked up in their shard at doc_offset; {bad} wrong"
        safe(r8, "every token line points at the right token in its shard (shard, doc, offset -> token id) and its ppl = "
                 "exp(loss)", lookup)
        samp = jl(L_main / "learning_samples.jsonl")
        agg = defaultdict(float)
        for row in rows_csv:
            agg[(int(row[ix["step"]]), int(row[ix["row"]]), int(row[ix["span_index"]]))] += float(row[ix["loss"]])
        sbad = [s_ for s_ in samp if s_["n_loss_tokens"] and abs(agg[(s_["step"], s_["row"], s_["span_index"])] - s_["loss_sum"]) > 1e-6]
        check(r8, "sample-level loss sums equal the sum of their token lines", samp and not sbad,
              f"{len(samp)} samples, {len(sbad)} mismatches")

        def learn_control():
            b = [list(x) for x in raw_lines[main_recs[3]["global_step"]]]
            b[0][ix["loss"]] = repr(float(b[0][ix["loss"]]) + 0.5)
            return trace_hash(b) != main_recs[3]["learning_trace_sha256"], "a faked token loss breaks the bound hash"
        safe(r8, "negative control: a faked token loss is detected", learn_control)
        ecks = jl(L_main / "eval_checkpoints.jsonl")
        lane_loss = defaultdict(lambda: [0.0, 0])
        for s_ in samp:
            if s_["n_loss_tokens"]:
                lane_loss[(s_["lane"], s_["phase"])][0] += s_["loss_sum"]
                lane_loss[(s_["lane"], s_["phase"])][1] += s_["n_loss_tokens"]
        check(r8, "loss is produced by execution (falls over training) and validation loss is tracked per language",
              main_recs[-1]["step_loss"] < main_recs[0]["step_loss"] and ecks and all(len(e["validation_loss"]) == 4 for e in ecks),
              f"step loss {main_recs[0]['step_loss']:.3f} -> {main_recs[-1]['step_loss']:.3f}; validation en "
              f"{ecks[0]['validation_loss']['en']['loss']:.3f} -> {ecks[-1]['validation_loss']['en']['loss']:.3f}")
        facts["learning"] = {"token_lines": len(rows_csv), "samples": len(samp),
                             "first_last_step_loss": [main_recs[0]["step_loss"], main_recs[-1]["step_loss"]],
                             "validation_by_checkpoint": {e["step"]: {k: round(v["loss"], 4) for k, v in e["validation_loss"].items()} for e in ecks},
                             "mean_loss_by_lane_phase": {f"{k[0]}/{k[1]}": v[0] / v[1] for k, v in sorted(lane_loss.items())}}

    # ===== R11 ledger integrity / R12 checkpoint binding ==================================================
    r11 = E.req("R11", "Consumption ledger integrity", "Consumption ledger integrity", "Hash-chained ledger",
                ["ledgers/main/consumption_ledger.jsonl"])
    with section(r11):
        check(r11, "main, reference and fork ledgers verify end to end (hash chain, offsets)", not (main_err or ref_err or fork_err),
              f"{len(main_recs)} / {len(ref_recs)} / {len(fork_recs)} records")
        fields = ["run_id", "branch_id", "global_step", "checkpoint_id_base", "rank", "microbatches", "batch_id", "batch_hash",
                  "batch_loss_mask_hash", "attention_policy", "position_policy", "stage", "tokenizer_id", "dataloader_version",
                  "schedule_sha256", "opus"]
        rowf = ["sample_id", "shard_ids", "spans", "lane", "loss_mask_hash", "opus_decision_id", "row_hash"]
        check(r11, "every record names run, branch, step, checkpoint, rank, microbatches, packed samples, shards, spans, "
                   "loss-mask hash, attention/position policy, lane, stage, tokenizer, dataloader version, OPUS decision",
              all(all(k in x for k in fields) and all(all(k in row for k in rowf) for row in rows_of(x)) for x in main_recs), "")
    # ===== R12 checkpoint binding =========================================================================
    r12 = E.req("R12", "Checkpoints tied to ledger offsets", "Checkpoint / ledger binding", "Checkpoint meta vs ledger",
                ["checkpoints/main/*/meta.json"])
    with section(r12):
        metas = [rj(c / "meta.json") for c in sorted((art / "checkpoints" / "main").iterdir())]
        cb = []
        for m in metas:
            recset = pre_recs if m["step"] <= ck_step else main_recs
            ok = (m["ledger_offset"] == m["step"] and recset[m["step"] - 1]["record_hash"] == m["ledger_head_hash"]
                  and all(sha256_file(art / "checkpoints" / "main" / f"step_{m['step']:05d}" / f) == h for f, h in m["files"].items()))
            nb = m.get("next_batch")
            if nb:
                ok &= by_step[m["step"] + 1]["batch_id"] == nb["batch_id"] and by_step[m["step"] + 1]["batch_hash"] == nb["batch_hash"]
            cb.append((m["checkpoint_id"], ok))
        check(r12, "every checkpoint: offset == step, head hash == the ledger record at that offset, files hash, and its "
                   "declared next batch is the batch the ledger shows next", metas and all(ok for _, ok in cb), json.dumps(cb))
        stj = rj(art / "checkpoints" / "main" / f"step_{ck_step:05d}" / "state.json")
        check(r12, "the checkpoint's data state is explicit (per-stream pass/index/offset, deferral queue) and hashes to meta",
              sha256_json(stj["dataloader"]) == ckm["dataloader_state_sha256"] and stj["dataloader"]["streams"],
              f"{len(stj['dataloader']['streams'])} streams")
        facts["checkpoints"] = [{"id": m["checkpoint_id"], "offset": m["ledger_offset"], "head": m["ledger_head_hash"][:16],
                                 "next": (m.get("next_batch") or {}).get("batch_id")} for m in metas]

    # ===== R13 fork =======================================================================================
    r13 = E.req("R13", "Forking from an earlier checkpoint", "Fork", "Fork point and divergence",
                [f"ledgers/{fkid}/fork_point.json", f"ledgers/{fkid}/consumption_ledger.jsonl", f"ledgers/{fkid}/fork_schedule.json"])
    with section(r13):
        pk = rj(art / "checkpoints" / "main" / f"step_{fpoint['parent_checkpoint_step']:05d}" / "meta.json")
        check(r13, "the fork's ledger chains from the parent checkpoint's ledger head (genesis = parent head at the offset)",
              not fork_err and fork_recs and fork_recs[0]["prev_record_hash"] == pk["ledger_head_hash"] ==
              pre_recs[pk["ledger_offset"] - 1]["record_hash"], f"{pk['checkpoint_id']} offset {pk['ledger_offset']}")
        check(r13, "new branch identity, inherited model/optimizer/data state equal the parent checkpoint",
              all(x["branch_id"] == fkid != "main" for x in fork_recs) and
              fpoint["inherited"]["model_sha256"] == pk["model_sha256"] and
              fpoint["inherited"]["dataloader_state_sha256"] == pk["dataloader_state_sha256"], fkid)
        ch = sha256_json([x["batch_id"] for x in main_recs[:fpoint["parent_checkpoint_step"]]])
        div = [x["global_step"] for x in fork_recs if x["batch_id"] != by_step[x["global_step"]]["batch_id"]]
        check(r13, "common history before the fork is the parent's; after it the stream diverges by design",
              ch == fpoint["common_history"]["batch_ids_digest"] and div and div[0] == fpoint["first_fork_step"],
              f"first divergent step {div[0] if div else None}; {len(div)}/{len(fork_recs)} fork steps differ")
        fsch = rj(fdir / "fork_schedule.json")
        fin = defaultdict(lambda: [0, 0])
        for x in fork_recs:
            for row in rows_of(x):
                fin[x["stage"]][1] += row["L"]
                fin[x["stage"]][0] += row["L"] if row["lane"] == "indic" else 0
        ffloor = {st["stage"]: st["stage_floor_pct"]["indic"] for st in fsch["steps"] if st["step"] >= fpoint["first_fork_step"]}
        check(r13, "the fork executes its own A5 arm (H1: 16% Indic where overridden) under the same floor rule",
              fin and all(v[0] / v[1] >= ffloor[k] / 100 - 1e-9 for k, v in fin.items())
              and fsch["overrides"]["from_step"] == fpoint["first_fork_step"],
              json.dumps({k: f"{100 * v[0] / v[1]:.2f}% (floor {ffloor[k]}%)" for k, v in fin.items()}))
        facts["fork"] = {"branch": fkid, "parent": pk["checkpoint_id"], "parent_offset": pk["ledger_offset"],
                         "parent_head": pk["ledger_head_hash"], "first_fork_step": fpoint["first_fork_step"],
                         "first_fork_batch": fork_recs[0]["batch_id"], "main_batch_same_step": by_step[fpoint["first_fork_step"]]["batch_id"],
                         "indic_share_pct": {k: round(100 * v[0] / v[1], 2) for k, v in fin.items()}}

    # ===== R09 throughput =================================================================================
    r9 = E.req("R09", "Packing utilization and useful loss-bearing tokens per second", "Throughput", "Performance report",
               ["performance.json", "ledgers/main/performance_steps.jsonl"])
    with section(r9):
        ps = {p["step"]: p for p in jl(L_main / "performance_steps.jsonl")}

        def perf_recompute(perf_doc):
            T = sum(ps[s]["t_step_total_s"] for s in steps)
            raw = {"pos": sum(x["token_positions"] for x in main_recs), "real": sum(x["n_real_tokens"] for x in main_recs),
                   "loss": sum(x["n_loss_tokens"] for x in main_recs)}
            mm = perf_doc["branches"]["main"]["metrics"]
            exp = {"packing_utilization": raw["real"] / raw["pos"], "loss_bearing_fraction": raw["loss"] / raw["pos"],
                   "useful_loss_bearing_tokens_per_s": raw["loss"] / T, "raw_token_positions_per_s": raw["pos"] / T,
                   "accepted_real_tokens_per_s": raw["real"] / T,
                   "opus_acceptance_rate": sum(ps[s]["accepted_rows"] for s in steps) / sum(ps[s]["candidates_scored"] for s in steps)}
            bad = {k: (mm[k], v) for k, v in exp.items() if abs(mm[k] - v) > 1e-9 * max(1, abs(v))}
            return not bad, f"recomputed {len(exp)} metrics from raw step timings and the ledger; mismatches {bad}"
        safe(r9, "headline metrics recompute from performance_steps.jsonl and the consumption ledger", lambda: perf_recompute(perf))

        def perf_control():
            b = copy.deepcopy(perf)
            b["branches"]["main"]["metrics"]["useful_loss_bearing_tokens_per_s"] *= 1.5
            return not perf_recompute(b)[0], "an inflated throughput number is caught"
        safe(r9, "negative control: an inflated throughput number fails", perf_control)
        h = perf["headline"]
        check(r9, "useful loss-bearing tokens/s, accepted tokens/s, utilization, padding and context-only shares are reported "
                  "with formulas; GPU metrics are null with a reason, not zero",
              h["useful_loss_bearing_tokens_per_s"] > 0 and perf["gpu_metrics"]["gpu_idle_time"] is None and
              abs(h["packing_utilization"] - (1 - h["padding_fraction"])) < 1e-12,
              f"util {h['packing_utilization']:.4f}, loss-bearing {h['loss_bearing_fraction']:.4f}, useful tok/s "
              f"{h['useful_loss_bearing_tokens_per_s']:.1f}")
        check(r9, "pad-only counterfactual on the same consumed documents is computed for every policy",
              all("pad_only" in v["counterfactual_utilization_for_same_documents"] for v in perf["branches"]["main"]["by_policy"].values()),
              "; ".join(f"{k}: {v['utilization']:.3f} vs pad-only {v['counterfactual_utilization_for_same_documents']['pad_only']:.3f}"
                        for k, v in perf["branches"]["main"]["by_policy"].items()))
        facts["performance"] = {"headline": h, "latency": perf["latency"], "opus_by_lane": perf["opus_by_lane"]}

    # ===== R14 upstream contracts =========================================================================
    r14 = E.req("R14", "Upstream A2 / A4 / A5 contracts reconciled", "Upstream contracts", "Contract audit",
                ["upstream_contracts.json", "inputs/PROVENANCE.json"])
    with section(r14):
        uc = rj(art / "upstream_contracts.json")
        prov = rj(REPO / "inputs" / "PROVENANCE.json")
        check(r14, "every vendored input still hashes to its PROVENANCE.json entry",
              all(sha256_file(REPO / "inputs" / k) == v["sha256"] for k, v in prov["files"].items()), f"{len(prov['files'])} files")
        code_files = list((REPO / "tdes").glob("*.py")) + [REPO / "run_demo.py"]
        pat = re.compile("|".join(["new" + r" ass [245]", "Down" + "loads"]))  # split so this line does not match itself
        hits = [p.name for p in code_files if pat.search(p.read_text(encoding="utf-8"))]
        check(r14, "runtime code never refers to the upstream assignment folders (only tools/prepare_inputs.py does)",
              not hits, f"{len(code_files)} files scanned; hits {hits}")
        check(r14, "no Llama-2-generated Anudesh response is vendored (prompts only)",
              all("messages" not in x for x in jl(REPO / "inputs" / "corpus" / "indic_anudesh_prompts.jsonl")), "")
        check(r14, "the contract record lists each discrepancy with a resolution",
              all(("resolution" in e) or e["result"] for e in uc["entries"]), f"{len(uc['entries'])} entries; "
              f"{sum(1 for e in uc['entries'] if not e['result'])} discrepancies")
        facts["upstream"] = [{k: e.get(k) for k in ("upstream", "artifact", "result", "discrepancy", "resolution")} for e in uc["entries"]]

    # ===== R15 end-to-end =================================================================================
    r15 = E.req("R15", "End-to-end execution", "End-to-end execution", "run.log event sequence", ["run.log"])
    with section(r15):
        log = (art / "run.log").read_text(encoding="utf-8")
        seq = ["shards created", "manifests validated", "evaluation data blocked", "mixture compiled", "batches packed",
               "OPUS decisions recorded", "checkpoint saved", "crash simulated", "run resumed", "historical stream replayed",
               "branch forked", "audit completed", "performance measured"]
        pos_ = [log.find(s) for s in seq]
        check(r15, "run.log contains the full event sequence in order", all(p >= 0 for p in pos_) and pos_ == sorted(pos_),
              ", ".join(f"{s}@{p}" for s, p in zip(seq, pos_)))
        flags = ["[PASS] tokenizer_hash_verified", "[PASS] eval_shard_blocked", "[PASS] checkpoint_saved",
                 "[PASS] resume_next_batch_matched", "[PASS] replay_hash_matched"]
        check(r15, "the required PASS events are in run.log and no phase wrote a FAIL",
              all(f in log for f in flags) and "[FAIL]" not in log, "; ".join(flags))
        req_paths = ["run.log", "manifests", "ledgers", "checkpoints", "performance.json"]
        check(r15, "the required artifact structure exists", all((art / p).exists() for p in req_paths), ", ".join(req_paths))

    # ===== write ==========================================================================================
    for q in E.reqs:
        q["result"] = "PASS" if q["checks"] and all(c["pass"] for c in q["checks"]) else "FAIL"
    overall = all(q["result"] == "PASS" for q in E.reqs)
    ev = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"), "generator": "tdes/verify.py",
          "overall": "PASS" if overall else "FAIL", "requirements": E.reqs, "key_facts": facts}
    (art / "evidence.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    (art / "evidence.md").write_text(render_md(ev), encoding="utf-8")
    if not a.no_log:
        from . import runlog
        for q in E.reqs:
            runlog.log(art, f"[{q['result']}] {q['id']} {q['name']} | {sum(c['pass'] for c in q['checks'])}/{len(q['checks'])} checks")
        runlog.log(art, f"evidence written: evidence.json, evidence.md; overall {ev['overall']}")
    print("overall", ev["overall"])
    for q in E.reqs:
        if q["result"] != "PASS":
            for c in q["checks"]:
                if not c["pass"]:
                    print("FAIL", q["id"], c["name"], "|", c["detail"][:400])
    return 0 if overall else 1


def render_md(ev):
    f = defaultdict(lambda: defaultdict(lambda: "n/a"), ev["key_facts"])
    by = {q["id"]: q for q in ev["requirements"]}
    main_ids = ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09"]
    out = ["# Evidence summary", "",
           f"Generated by `tdes/verify.py` from the files in `artifacts/` ({ev['generated_at']}). "
           f"Overall: **{ev['overall']}**. Every row is derived from re-computation, not from flags written by other phases.", "",
           "| Requirement | Result | Evidence |", "| --- | --- | --- |"]
    for i in main_ids:
        q = by[i]
        out.append(f"| {q['table_name']} | {q['result']} | {q['evidence_label']}: " + ", ".join(f"`{x}`" for x in q["files"]) + " |")
    out += ["", "Additional requirements:", "", "| Requirement | Result | Evidence |", "| --- | --- | --- |"]
    for q in ev["requirements"]:
        if q["id"] not in main_ids:
            out.append(f"| {q['table_name']} | {q['result']} | {q['evidence_label']}: " + ", ".join(f"`{x}`" for x in q["files"]) + " |")
    t, c = f["tokenizer"], f["crash"]
    out += ["", "## Key identities", "",
            f"- Tokenizer: A2 `tokenizer.json` sha256 `{t['a2_sha256'][:16]}`, adapter tokenizer_id `{t['tokenizer_id'][:16]}` "
            f"({t['a2_vocab']} A2 tokens + {len(t['control_tokens'])} control tokens).",
            f"- Crash: step {c['crash_step']}, exit code {c['exit_code']}, {c['torn_tail_bytes']} torn bytes; recovered to "
            f"`{c['checkpoint']}` (ledger offset {c['ledger_offset']}); orphaned steps {c['orphaned']}.",
            f"- Expected next batch (declared by the checkpoint): `{c['expected_next_batch']}` / `{c['expected_hash'][:16]}`; "
            f"first resumed batch: `{c['resumed_batch']}` / `{c['resumed_hash'][:16]}`.",
            f"- Replay steps {f['replay']['interval'][0]}-{f['replay']['interval'][1]} from `{f['replay']['from']}`: stream "
            f"fingerprint `{f['replay']['fingerprint'][:16]}` (original and replay).",
            f"- Fork `{f['fork']['branch']}` from `{f['fork']['parent']}` (offset {f['fork']['parent_offset']}): first fork "
            f"batch `{f['fork']['first_fork_batch']}` vs main `{f['fork']['main_batch_same_step']}`.",
            f"- OPUS: {json.dumps(f['opus']['counts'])}.",
            "", "## Planned versus actual lane shares (% of token positions per stage)", "",
            "| Stage | " + " | ".join(next(iter(f["mixture"]["stage_shares"].values())).keys()) + " |",
            "| --- |" + " --- |" * len(next(iter(f["mixture"]["stage_shares"].values())))]
    for st, v in f["mixture"]["stage_shares"].items():
        out.append(f"| {st} | " + " | ".join(f"{x['planned_pct_incl_ramps']:.1f} / {x['actual_pct']:.1f}" for x in v.values()) + " |")
    out += ["", "Cells are planned / actual. Planned includes the ramps; protected lanes (indic, reasoning, agentic) round "
                "up to whole rows, so at demo scale a lane with a 0.5% share still gets one full row.", "",
            "## Checks behind each row", ""]
    for q in ev["requirements"]:
        out.append(f"**{q['id']} {q['name']}: {q['result']}**")
        for ch in q["checks"]:
            d = ch["detail"]
            out.append(f"- {'PASS' if ch['pass'] else 'FAIL'}: {ch['name']}" + (f" ({d[:300]})" if d else ""))
        out.append("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
