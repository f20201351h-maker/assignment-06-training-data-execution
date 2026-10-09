"""Upstream contracts: load the A5 mixture plan and scale it to the demo, check A2/A4 identities.

`load_a5_plan` turns A5's `mixture.yaml` (token budgets in billions) into a demo-scale plan:
  * stage lengths in optimizer steps from A5's `span` (largest remainder over `main_steps`);
  * anneal length from A5's anneal/main ratio, raised to `anneal_steps_min` (the distortion is recorded);
  * window lengths = A5 seq_len / `seq_len_divisor` (4096 -> 128, 8192 -> 256; long-context 32768 -> 1024);
  * ramp width = A5 transition_ramp_tokens scaled to steps, raised to `ramp_steps_min` (recorded);
  * lane mixes, protected lanes, selector keep fractions, reserve fractions and max passes, unchanged.
Every scaling decision is written into the plan under `scaling_notes` so the README and the verifier can
quote real numbers instead of prose.
"""
import json
import math
import re
from pathlib import Path

import yaml

from .hashing import sha256_file, sha256_json

LONG_LANES = ("agentic", "long_context")


def largest_remainder(weights, total):
    raw = {k: w * total / sum(weights.values()) for k, w in weights.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), list(raw).index(k)))[:total - sum(out.values())]:
        out[k] += 1
    return out


def load_a5_plan(cfg, repo):
    path = Path(repo) / cfg["mixture_contract"]
    a5 = yaml.safe_load(path.read_text(encoding="utf-8"))
    ds = cfg["demo_scale"]
    lanes = list(a5["lanes"])
    stage_names = list(a5["stages"])
    n_main = ds["main_steps"]
    steps = largest_remainder({s: a5["stages"][s]["span"] for s in stage_names}, n_main)
    anneal_exact = n_main * a5["budget"]["anneal"] / a5["budget"]["main_run"]
    n_anneal = max(ds["anneal_steps_min"], round(anneal_exact))
    stages, start = [], 1
    for s in stage_names + ["anneal"]:
        n = steps[s] if s != "anneal" else n_anneal
        src = a5["stages"][s] if s != "anneal" else None
        if s == "anneal":
            seq = stages[-1]["a5_seq_len"]  # A5: "same as S4"
            mix = a5["anneal_mix"]
        else:
            seq = src["seq_len"] if isinstance(src["seq_len"], int) else int(re.match(r"\d+", str(src["seq_len"])).group())
            mix = src["mix"]
        stages.append({"name": s, "step_start": start, "step_end": start + n - 1, "n_steps": n, "a5_seq_len": seq,
                       "seq_len": seq // ds["seq_len_divisor"], "mix": {l: float(mix[l]) for l in lanes},
                       "a5_span": src["span"] if src else None})
        start += n
    ramp_exact = a5["transition_ramp_tokens"] / a5["budget"]["main_run"] * n_main
    ramp = max(ds["ramp_steps_min"], round(ramp_exact))
    reserve = {}
    for lane, pool in [("web", "web"), ("code", "code"), ("stem", "stem"), ("long_context", "long_context"),
                       ("reasoning", "reasoning")]:
        p = a5["pools"][pool]
        reserve[lane] = p["reserve"] / p["unique"]
    plan = {
        "source": cfg["mixture_contract"], "source_sha256": sha256_file(path),
        "lanes": lanes, "stages": stages, "total_steps": start - 1,
        "protected": list(a5["protected"]), "selector_keep": dict(a5["selector_keep"]),
        "selector_bypass": [l for l in lanes if l not in a5["protected"] and l not in a5["selector_keep"]],
        "ramp_steps": ramp, "max_passes": a5["max_passes"], "reserve_fraction": reserve,
        "indic_main_tiers": a5["indic_main_tiers"], "indic_anneal": a5["indic_anneal"],
        "agentic_pools": {k: a5["pools"][k] for k in ("agentic_tierA", "agentic_multistep", "agentic_oneshot")},
        "regular_positions_per_step": ds["regular_positions_per_step"], "long_row_len": ds["long_row_len"],
        "microbatch_positions": ds["microbatch_positions"],
        "scaling_notes": {
            "stage_steps": {s["name"]: s["n_steps"] for s in stages},
            "anneal_steps_exact": anneal_exact, "anneal_steps_used": n_anneal,
            "anneal_share_of_trained_planned": a5["budget"]["anneal"] / (a5["budget"]["main_run"] + a5["budget"]["anneal"]),
            "anneal_share_of_steps_used": n_anneal / (start - 1),
            "ramp_steps_exact": ramp_exact, "ramp_steps_used": ramp,
            "seq_len_divisor": ds["seq_len_divisor"],
            "long_row_len": f"A5 32768-token long-context batches / {ds['seq_len_divisor']} = {ds['long_row_len']}",
            "agentic_rows": "A5 notes trajectories do not fit the base window; at demo scale agentic rows use the "
                            "long-row microbatch (whole trajectories, never split)",
        },
    }
    plan["plan_sha256"] = sha256_json(plan)
    return plan


def stage_of(plan, step):
    for i, st in enumerate(plan["stages"]):
        if st["step_start"] <= step <= st["step_end"]:
            return i, st
    raise ValueError(f"step {step} outside the plan")


def a2_contract(repo, cfg, tok):
    fc = json.loads((Path(repo) / "inputs" / "a2" / "final_config.json").read_text(encoding="utf-8"))
    return {"a2_published_tokenizer_sha256": fc["tokenizer_sha256"], "vendored_file_sha256": tok.a2_sha256,
            "pinned_in_config": cfg["tokenizer"]["a2_sha256"], "a2_languages": fc["langs"],
            "a2_vocab_size": fc["config"]["vocab_size"], "a2_normalization": fc["config"]["norm"],
            "a2_byte_fallback": fc["config"]["byte_fallback"], "a2_composition": fc["composition"],
            "tokenizer_id": tok.tokenizer_id, "control_tokens": tok.special_ids}


def a4_contract(repo):
    m = json.loads((Path(repo) / "inputs" / "a4" / "manifest.json").read_text(encoding="utf-8"))
    out = {"corpus_id": m["corpus_id"], "shards": {}}
    for s in m["shards"]:
        name = "glaive" if "glaive" in s["source"] else "anudesh"
        out["shards"][name] = {
            "a4_shard_id": s["shard_id"], "content_sha256": s["content_sha256"], "license": s["license"],
            "terms_note": s["terms_note"], "status": s["status"], "docs": s["docs"], "a4_token_count": s["token_count"],
            "a4_tokenizer": s["tokenizer"], "languages": s["languages"], "config_sha256": s["config_sha256"],
            "cleaning_pipeline_id": sha256_json({"config": s["config_sha256"],
                                                 "scripts": [[c["name"], c["sha256"]] for c in s["cleaning_scripts"]]})}
    return out
