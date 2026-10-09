"""Build phase: inputs -> ingress gate -> tokenized shards -> manifests -> registries -> compiled A5 schedule.

    python -m tdes.build --artifacts artifacts
"""
import argparse
import copy
import json
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from . import runlog
from .context import REPO, load_config, load_tokenizer
from .contracts import a2_contract, a4_contract, load_a5_plan
from .corpus import admit, assign_lanes, ingress_hash, load_inputs
from .firewall import Firewall, build_eval_registry
from .hashing import sha256_file, sha256_json
from .schedule import compile_quotas, compile_schedule, lane_demand_positions
from .shards import (ImmutableShardError, ShardIntegrityError, ShardStore, admission_gate, build_all_shards,
                     content_hash, store_shard_bytes, write_registry)
from .tokenizer import A2Tokenizer


def _dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def build_shards(art, cfg, tok, plan, quotas):
    """Everything from vendored inputs to the shard registry. Deterministic; run twice to prove it."""
    docs, evals, prov = load_inputs(REPO, cfg)
    admitted, evals, report = admit(docs, evals, tok, cfg, drill=cfg["firewall_drills"]["ingress_contaminated_document"])
    anneal = lane_demand_positions(quotas, plan["lanes"], lambda s: s == "anneal")
    admitted, unplaced, reserve_notes = assign_lanes(admitted, tok, cfg, plan, anneal)
    a4 = a4_contract(REPO)
    common = {"transform_id": ingress_hash(cfg),
              "a4_shard_ids": {f"a4:{k}": v["a4_shard_id"] for k, v in a4["shards"].items()},
              "a4_pipeline_ids": {f"a4:{k}": v["cleaning_pipeline_id"] for k, v in a4["shards"].items()}}
    manifests = build_all_shards(art, admitted, evals, tok, common)
    reg = write_registry(art, manifests, tok.tokenizer_id)
    return {"docs": docs, "evals": evals, "prov": prov, "admitted": admitted, "report": report, "unplaced": unplaced,
            "reserve_notes": reserve_notes, "manifests": manifests, "registry": reg, "common": common, "a4": a4}


def validate_manifests(art, cfg, tok, admitted_by_id, eval_by_id, lineages):
    """Re-open every shard from disk, recompute hashes, re-encode its documents, run the admission gate."""
    store = ShardStore(art, tok.tokenizer_id)
    out, all_ok = [], True
    for sid in sorted(store.registry):
        m = store.manifest(sid)
        t, e = store.arrays(sid)
        files_ok = all(sha256_file(art / "shards" / sid / f) == v["sha256"] for f, v in m["files"].items())
        re_t, re_e = [], []
        for di in m["doc_index"]:
            ids, el = tok.render_document(admitted_by_id.get(di["doc_id"]) or eval_by_id[di["doc_id"]])
            re_t += ids
            re_e += el
        reencode_ok = np.array_equal(np.asarray(re_t), t) and np.array_equal(np.asarray(re_e), e)
        recomputed = content_hash(tok.tokenizer_id, m["transform_id"], np.asarray(re_t), np.asarray(re_e), m["doc_index"])
        gate = admission_gate(m, tok.tokenizer_id, lineages, cfg["cleaning"]["allowed_licenses"],
                              purpose="train" if m["split"] == "train" else "eval")
        ok = files_ok and reencode_ok and recomputed == m["content_hash"] and sid.endswith(recomputed[:12]) and not gate
        all_ok &= ok
        out.append({"shard_id": sid, "split": m["split"], "lane": m["lane"], "pool": m["pool"], "subpool": m["subpool"],
                    "manifest_content_hash": m["content_hash"], "recomputed_content_hash": recomputed,
                    "file_hashes_match": files_ok, "reencoded_from_documents_match": reencode_ok,
                    "tokenizer_id": m["tokenizer_id"], "admission_gate_reasons": gate, "valid": ok})
    return all_ok, out


def integrity_probes(art, store, tok, cfg):
    """(1) rewrite a registered shard with different bytes: the writer must refuse;
    (2) tamper with a COPY of a shard: the validator must reject it under its old identity;
    (3) a tokenizer that differs only in its control-token block: the store must refuse every shard."""
    sid = store.shards_where(split="train")[0]
    sdir = art / "shards" / sid
    tb, eb = (sdir / "tokens.bin").read_bytes(), (sdir / "loss_eligible.bin").read_bytes()
    same = store_shard_bytes(sdir, tb, eb)
    try:
        store_shard_bytes(sdir, bytes([tb[0] ^ 1]) + tb[1:], eb)
        refused, msg = False, None
    except ImmutableShardError as ex:
        refused, msg = True, str(ex)
    unchanged = sha256_file(sdir / "tokens.bin") == store.manifest(sid)["files"]["tokens.bin"]["sha256"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "manifests").mkdir()
        shutil.copytree(art / "manifests" / "shards", td / "manifests" / "shards")
        shutil.copyfile(art / "manifests" / "shard_registry.json", td / "manifests" / "shard_registry.json")
        (td / "shards").mkdir()
        shutil.copytree(sdir, td / "shards" / sid)
        tp = td / "shards" / sid / "tokens.bin"
        tp.chmod(0o666)
        raw = bytearray(tp.read_bytes())
        raw[10] ^= 0x01
        tp.write_bytes(bytes(raw))
        try:
            ShardStore(td, tok.tokenizer_id).arrays(sid)
            tamper_caught, tmsg = False, None
        except ShardIntegrityError as ex:
            tamper_caught, tmsg = True, str(ex)
    other = copy.copy(tok)
    swapped = dict(tok.special_ids, **{"<|pad|>": tok.eos_id, "<|eos|>": tok.pad_id})  # EOS and PAD ids swapped
    spec = dict(tok.spec, control_tokens=swapped)
    other.tokenizer_id = sha256_json(spec)
    try:
        ShardStore(art, other.tokenizer_id)
        mismatch_refused, mmsg = False, None
    except ShardIntegrityError as ex:
        mismatch_refused, mmsg = True, str(ex)
    return {"immutability": {"shard_id": sid, "identical_rewrite": same, "different_rewrite_refused": refused,
                             "message": msg, "shard_unchanged_after_probe": unchanged, "pass": refused and unchanged},
            "tamper": {"shard_id": sid, "byte_flipped": 10, "copy_rejected": tamper_caught, "message": tmsg,
                       "pass": tamper_caught},
            "tokenizer_mismatch": {"variant": "EOS and PAD ids swapped", "variant_tokenizer_id": other.tokenizer_id,
                                   "store_refused": mismatch_refused, "message": mmsg, "pass": mismatch_refused}}


def upstream_contracts(art, cfg, tok, plan, b, sched):
    """Machine-readable reconciliation of A2 / A4 / A5 with what this run actually built."""
    a2 = a2_contract(REPO, cfg, tok)
    rep = b["report"]
    cov = Counter((r["language"], r["script"]) for r in rep if any(x.startswith("a2_tokenizer_coverage")
                                                                     for x in r.get("reasons", [])))
    adm = b["admitted"]
    by = lambda f: dict(sorted(Counter(f(d) for d in adm).items()))
    lanes_supplied = {l: sorted({d["source"] for d in adm if d["lane"] == l}) for l in plan["lanes"]}
    indic_tok = sum(d["n_tokens"] for d in adm if d["lane"] == "indic")
    tiers = plan["indic_main_tiers"]
    vend = b["prov"]["files"]
    entries = [
        {"upstream": "A2", "artifact": "inputs/a2/tokenizer.json", "identity": a2["vendored_file_sha256"],
         "check": "vendored file sha256 == hash A2 published in final_config.json == pinned in config",
         "result": a2["vendored_file_sha256"] == a2["a2_published_tokenizer_sha256"] == a2["pinned_in_config"],
         "consumer": "tdes.tokenizer.A2Tokenizer (every shard and ledger record carries tokenizer_id)"},
        {"upstream": "A2", "artifact": "special tokens", "identity": a2["tokenizer_id"],
         "check": "A2 defines EOS/PAD/role tokens", "result": False,
         "discrepancy": f"A2 ships only <unk> ({a2['a2_composition']}); packing needs EOS, PAD and role markers",
         "resolution": "control-token block appended after the A2 vocabulary (ids 10000+), never produced from "
                       "text; it is part of tokenizer_id", "consumer": "tdes.tokenizer.render_document"},
        {"upstream": "A2", "artifact": "normalizer", "check": "A2 normalisation preserves newlines/indentation",
         "result": tok.normalize("a\n    b") == "a\n    b",
         "discrepancy": "A2 collapses every whitespace run to one space (built for prose fertility), so code "
                        "loses line breaks and indentation",
         "resolution": "kept A2's mapping unchanged (changing it would be a different tokenizer); code structure is "
                       "preserved by whole-function packing; flagged for a code-aware A2 revision",
         "consumer": "code lane"},
        {"upstream": "A2 x A4", "artifact": "language coverage",
         "check": "every A4-admitted Indic language is encodable by A2 (unk rate <= max_unk_rate)",
         "result": not cov, "observed_rejections_by_language_script": {f"{k[0]}/{k[1]}": v for k, v in sorted(cov.items())},
         "discrepancy": "A4 admitted 15 languages; A2 was trained for en/hi/te/ur",
         "resolution": "A6 ingress gate drops documents above the unk threshold and logs them", "consumer": "tdes.corpus.admit"},
        {"upstream": "A4", "artifact": "inputs/a4/manifest.json",
         "identity": {k: v["a4_shard_id"] for k, v in b["a4"]["shards"].items()},
         "check": "vendored A4 records are recorded as parent shards with A4 cleaning lineage",
         "result": all(m["parent_shard_ids"] for m in b["manifests"] if m["a4_admitted"]),
         "imported_contract": {k: {"license": v["license"], "terms_note": v["terms_note"], "status": v["status"],
                                   "a4_tokenizer": v["a4_tokenizer"]} for k, v in b["a4"]["shards"].items()},
         "consumer": "manifests/shards/*.json (parent_shard_ids, cleaning_lineage)"},
        {"upstream": "A4", "artifact": "Anudesh licence", "check": "no Llama-2-generated response is vendored",
         "result": all(r["source"] != "a4:anudesh" or "messages" not in r for r in b["docs"]),
         "discrepancy": "A4 notes the Anudesh responses are Llama-2-70B-Chat outputs whose licence restricts "
                        "training other models; A4 did not publish them",
         "resolution": "only the human-written user prompts (CC-BY-4.0) are vendored; they form the Indic lane",
         "consumer": "indic lane"},
        {"upstream": "A4", "artifact": "token counts", "check": "A4 token counts are in A2 units",
         "result": False, "discrepancy": "A4 counted with Qwen/Qwen3-0.6B; A6 recounts every document with A2",
         "resolution": "manifests and the schedule use A2 token counts only", "a6_indic_tokens_A2": indic_tok,
         "consumer": "manifests, scarcity check"},
        {"upstream": "A4 x A5", "artifact": "lane supply", "check": "A4 supplies every A5 lane",
         "result": all(any(s.startswith("a4:") for s in v) for v in lanes_supplied.values()),
         "lanes_supplied_by": lanes_supplied,
         "discrepancy": "A4 cleaned two SFT shards (Glaive, Anudesh); A5 needs web, code, stem, reasoning, "
                        "long-context too",
         "resolution": "A2 Wikipedia text (web, stem, long-context), CPython stdlib functions (code) and generated "
                       "traces (reasoning), each labelled in its manifest provenance; they pass the A6 ingress "
                       "gate but are marked a4_admitted=false", "consumer": "tdes.corpus"},
        {"upstream": "A5", "artifact": "inputs/a5/mixture.yaml", "identity": plan["source_sha256"],
         "check": "stage mixes sum to 100 and the plan compiles", "result": True,
         "imported_contract": {"stages": {s["name"]: s["mix"] for s in plan["stages"]}, "protected": plan["protected"],
                               "selector_keep": plan["selector_keep"], "max_passes": plan["max_passes"]},
         "scaling": plan["scaling_notes"], "consumer": "tdes.schedule.compile_quotas"},
        {"upstream": "A5 x A4", "artifact": "Indic tiers", "check": "A4 supplies A5 Indic tiers A/B/C/D",
         "result": False, "a5_main_tier_tokens_B": tiers,
         "a6_supply_by_tier": by(lambda d: d["a5_tier"] if d["lane"] == "indic" else "-"),
         "discrepancy": "A5 plans 41.7% Tier A verified native; A4 supplied only Anudesh, which A5 classes as Tier D",
         "resolution": "the lane keeps its protected 10% share (A5: Indic is fixed by the floor) and is served from "
                       "Tier D; the tier split is reported, not faked", "consumer": "indic lane"},
        {"upstream": "A5 x A4", "artifact": "agentic tiers", "check": "agentic Tier A (executed, verified) exists",
         "result": False, "a5_pools": plan["agentic_pools"],
         "a6_supply": by(lambda d: d["a5_tier"] if d["lane"] == "agentic" else "-"),
         "resolution": "main run uses Glaive conversations with >=2 tool calls (multistep pool); the anneal uses the "
                       "one-shot Glaive pool, which A5 reserves in full", "consumer": "agentic lane"},
        {"upstream": "A5", "artifact": "reasoning length bands",
         "check": "every gated band has supply", "result": not sched["band_substitutions"],
         "substitutions": sched["band_substitutions"],
         "resolution": "a band without supply is served by the nearest band that has it, and each substitution is "
                       "listed", "consumer": "tdes.schedule.compile_schedule"},
    ]
    out = {"format": "tdes-upstream-contracts/1", "entries": entries, "reserve_sizing": b["reserve_notes"],
           "vendored_inputs": {k: {"sha256": v["sha256"], "source": v["source"], "licence": v["licence"]}
                               for k, v in sorted(vend.items())}}
    _dump(art / "upstream_contracts.json", out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    art = Path(a.artifacts)
    cfg = load_config(a.config)
    tok = load_tokenizer(cfg)  # raises TokenizerHashMismatch if the A2 file or the adapter changed
    a2 = a2_contract(REPO, cfg, tok)
    tok_ok = (tok.a2_sha256 == cfg["tokenizer"]["a2_sha256"] == a2["a2_published_tokenizer_sha256"]
              and tok.tokenizer_id == cfg["tokenizer"]["tokenizer_id"])
    probe = "भारत एक देश है। Hello  world"
    _dump(art / "manifests" / "tokenizer_manifest.json", {
        "a2_file": cfg["tokenizer"]["path"], "a2_file_sha256": tok.a2_sha256,
        "a2_published_sha256": a2["a2_published_tokenizer_sha256"], "tokenizer_id": tok.tokenizer_id,
        "pinned": cfg["tokenizer"], "spec": tok.spec, "vocab_size": tok.vocab_size,
        "unicodedata_version": unicodedata.unidata_version, "frozen": True,
        "determinism_probe": {"text": probe, "ids": tok.encode(probe), "decoded": tok.decode(tok.encode(probe)),
                              "normalized": tok.normalize(probe)}})
    runlog.check(art, "tokenizer_hash_verified", tok_ok,
                 f"A2 tokenizer.json sha256 {tok.a2_sha256[:16]} == A2-published == pinned; adapter tokenizer_id "
                 f"{tok.tokenizer_id[:16]}; vocab {tok.base_vocab}+{len(tok.special_ids)} control tokens")
    if not tok_ok:
        sys.exit(2)
    plan = load_a5_plan(cfg, REPO)
    _dump(art / "manifests" / "a5_plan.json", plan)
    quotas = compile_quotas(plan, cfg)
    b = build_shards(art, cfg, tok, plan, quotas)
    rep = b["report"]
    dropped = [r for r in rep if r["decision"] == "dropped"]
    _dump(art / "manifests" / "admission_report.json", {
        "ingress_transform_id": b["common"]["transform_id"], "decisions": rep,
        "unplaced_too_long": b["unplaced"], "reserve_sizing": b["reserve_notes"],
        "summary": dict(Counter(r["decision"] for r in rep)),
        "dropped_by_reason": dict(Counter(r["reasons"][0].split(":")[0] for r in dropped))})
    runlog.log(art, f"documents: {len(b['docs'])} training candidates, {len(b['admitted'])} admitted, {len(dropped)} "
                    f"dropped by the A6 ingress gate ({dict(Counter(r['reasons'][0].split(':')[0] for r in dropped))})")
    drill = [r for r in rep if r.get("drill")]
    runlog.log(art, f"ingress gate: eval-contaminated drill document {drill[0]['doc_id']} dropped -> {drill[0]['reasons']}"
               if drill else "ingress drill missing")
    by = Counter(f"{m['split']}/{m['lane']}/{m['pool']}" for m in b["manifests"])
    runlog.log(art, f"shards created: {len(b['manifests'])} immutable shards, "
                    f"{sum(m['token_count'] for m in b['manifests'])} tokens; {dict(sorted(by.items()))}")
    with tempfile.TemporaryDirectory() as td:  # same inputs + tokenizer + config, built again from scratch
        b2 = build_shards(Path(td), cfg, tok, plan, quotas)
        det = {"first_registry_sha256": b["registry"]["registry_sha256"],
               "second_registry_sha256": b2["registry"]["registry_sha256"],
               "same_shard_ids": sorted(m["shard_id"] for m in b["manifests"]) == sorted(m["shard_id"] for m in b2["manifests"])}
        det["pass"] = det["same_shard_ids"] and det["first_registry_sha256"] == det["second_registry_sha256"]
    runlog.check(art, "shard_build_deterministic", det["pass"],
                 f"second independent build: registry {det['second_registry_sha256'][:16]} == {det['first_registry_sha256'][:16]}")
    store = ShardStore(art, tok.tokenizer_id)
    probes = integrity_probes(art, store, tok, cfg)
    runlog.check(art, "shard_immutability_enforced", probes["immutability"]["pass"], probes["immutability"]["message"] or "")
    runlog.check(art, "tampered_shard_rejected", probes["tamper"]["pass"], probes["tamper"]["message"] or "")
    runlog.check(art, "tokenizer_mismatch_rejected", probes["tokenizer_mismatch"]["pass"],
                 probes["tokenizer_mismatch"]["message"] or "")
    lineages = set(b["common"]["a4_pipeline_ids"].values()) | {b["common"]["transform_id"]}
    ok, val = validate_manifests(art, cfg, tok, {d["doc_id"]: d for d in b["admitted"]},
                                 {d["doc_id"]: d for d in b["evals"]}, lineages)
    _dump(art / "manifests" / "manifest_validation.json", {"all_valid": ok, "determinism": det, "probes": probes,
                                                            "known_cleaning_lineages": sorted(lineages), "shards": val})
    runlog.log(art, f"manifests validated: {sum(v['valid'] for v in val)}/{len(val)} shards valid (content hash "
                    f"recomputed, files re-hashed, documents re-encoded with A2, admission gate)")
    runlog.check(art, "manifests_validated", ok)
    reg = build_eval_registry(art, [m for m in b["manifests"] if m["split"] != "train"], b["evals"],
                              cfg["cleaning"]["fingerprint_ngram_words"])
    store = ShardStore(art, tok.tokenizer_id)
    fw = Firewall(art, store, tok, art / "ledgers" / "build" / "firewall_events.jsonl")
    sched = compile_schedule(plan, cfg, store, fw, cfg["firewall_drills"]["compile_time_misconfigured_sources"],
                             quotas=quotas)
    blocked = sched["blocked_sources"]
    srcs = {s for l in sched["lanes"].values() for p in l["sources"].values() for v in p.values() for s in v}
    runlog.log(art, f"evaluation data blocked: the mixture compiler was handed a misconfigured web source list "
                    f"{cfg['firewall_drills']['compile_time_misconfigured_sources']}; {len(blocked)} shard(s) removed: "
                    + "; ".join(f"{x['shard_id']} ({', '.join(x['reasons'])})" for x in blocked))
    runlog.check(art, "eval_shard_blocked",
                 {store.registry[x["shard_id"]]["split"] for x in blocked} >= {"test", "validation"}
                 and not any(x["shard_id"] in srcs for x in blocked) and bool(drill) and drill[0]["decision"] == "dropped",
                 f"ingress drill dropped; compiler blocked {len(blocked)}; eval registry {reg['registry_sha256'][:16]}")
    _dump(art / "manifests" / "mixture_schedule.json", sched)
    upstream_contracts(art, cfg, tok, plan, b, sched)
    sc = ", ".join(f"{s['lane']}/{s['pool']}/{s['subpool']}:{s['passes_needed']}" for s in sched["scarcity"])
    runlog.log(art, f"mixture compiled: A5 {Path(cfg['mixture_contract']).name} ({plan['source_sha256'][:12]}) -> "
                    f"{plan['total_steps']} steps, stages {plan['scaling_notes']['stage_steps']}, ramp "
                    f"{plan['ramp_steps']} steps, schedule {sched['schedule_sha256'][:16]}; passes needed {sc}; "
                    f"band substitutions {len(sched['band_substitutions'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
