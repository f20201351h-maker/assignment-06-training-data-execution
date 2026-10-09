"""Compile the A5 curriculum into per-step, per-lane row quotas (the executable mixture timeline).

Phase 1, `compile_quotas(plan, cfg)`, is a pure function of the A5 plan and the demo scale:

  * effective mixture of step s: the stage mix, or inside a ramp the linear blend
    (1 - a) * mix_prev + a * mix_next with a = (s - ramp_start + 1) / (w + 1)  (A5: ramps centred on the
    boundary, never a hard step);
  * base window: the stage seq_len, moved `window_moves_steps_before_ramp` steps before a ramp when the
    window changes (A5: "the window and the mixture should not change at the same step; move the window first");
  * positions: every step has a fixed regular microbatch budget P (rows of the base window); the long lanes
    (agentic, long_context) get extra 1024-position rows so that their share of ALL positions equals A5's
    share: target_l = P * m_l / (1 - m_agentic - m_long_context);
  * rounding is cumulative inside a segment (a stage, or a fork's overridden part of one):
      - protected lanes (A5 `protected`): the floor is the lane's share in the step's OWN stage (A5's floor
        table), measured against the positions ACTUALLY consumed in the segment so far (long rows included),
        rounded up because a floor is a minimum; anything a ramp adds on top is rounded to the nearest row.
        So at every step the cumulative share of each protected lane is >= its stage floor;
      - long_context (selector bypass, not a floor): rounded to the nearest whole long row;
      - unprotected regular lanes share the remaining base rows by largest remainder of their deficits.
Phase 2, `compile_schedule(...)`, binds quotas to admitted shards: per-lane pools (main / anneal reserve),
sub-pool splits (Indic difficulty bands, reasoning length bands), the eval firewall over every source list,
and the supply check (passes needed vs A5's max_passes).
"""
import copy
import math

from .contracts import LONG_LANES, stage_of
from .hashing import sha256_json

EPS = 1e-9


class ScheduleError(Exception):
    pass


def _stage_mix(plan, k, step, overrides):
    st = plan["stages"][k]
    mix = dict(st["mix"])
    if overrides and step >= overrides["from_step"] and st["name"] in overrides["mix"]:
        mix.update(overrides["mix"][st["name"]])
    if abs(sum(mix.values()) - 100) > 1e-6:
        raise ScheduleError(f"{st['name']} mix sums to {sum(mix.values())}")
    return mix


def _ramp(plan, step):
    """(k_prev, k_next, alpha) if `step` lies in a ramp, else None."""
    w = plan["ramp_steps"]
    for k in range(1, len(plan["stages"])):
        B = plan["stages"][k]["step_start"]
        lo = B - w // 2
        if lo <= step <= lo + w - 1:
            return k - 1, k, (step - lo + 1) / (w + 1)
    return None


def _window(plan, cfg, step):
    k, st = stage_of(plan, step)
    lead = cfg["demo_scale"]["window_moves_steps_before_ramp"]
    w = plan["ramp_steps"]
    if k + 1 < len(plan["stages"]):
        nxt = plan["stages"][k + 1]
        if nxt["seq_len"] != st["seq_len"] and step >= nxt["step_start"] - w // 2 - lead:
            return nxt["seq_len"]
    return st["seq_len"]


def effective_mix(plan, step, overrides=None):
    k, st = stage_of(plan, step)
    r = _ramp(plan, step)
    if r is None:
        return _stage_mix(plan, k, step, overrides), None
    kp, kn, a = r
    mp, mn = _stage_mix(plan, kp, step, overrides), _stage_mix(plan, kn, step, overrides)
    return {l: (1 - a) * mp[l] + a * mn[l] for l in plan["lanes"]}, {
        "from": plan["stages"][kp]["name"], "to": plan["stages"][kn]["name"], "alpha": a,
        "mix_from": mp, "mix_to": mn}


def compile_quotas(plan, cfg, overrides=None):
    lanes = plan["lanes"]
    P, LL = plan["regular_positions_per_step"], plan["long_row_len"]
    prot = set(plan["protected"])
    out, seg_key, cum_t, cum_a, cum_total = [], None, {}, {}, 0
    for s in range(1, plan["total_steps"] + 1):
        k, st = stage_of(plan, s)
        key = (st["name"], bool(overrides and s >= overrides["from_step"]))
        if key != seg_key:  # a new accounting segment: stage boundary, or the start of a fork's override
            seg_key, cum_t, cum_a, cum_total = key, {l: 0.0 for l in lanes}, {l: 0 for l in lanes}, 0
        mix, ramp = effective_mix(plan, s, overrides)
        m = {l: mix[l] / 100.0 for l in lanes}
        L = _window(plan, cfg, s)
        R = P // L
        mlong = sum(m[l] for l in LONG_LANES)
        row_len = {l: (LL if cfg["lanes"][l]["row"] == "long" else L) for l in lanes}
        tgt = {l: P * m[l] / (1 - mlong) for l in lanes}
        own = _stage_mix(plan, k, s, overrides)  # the floor of a protected lane is its share in its own stage
        for l in lanes:
            cum_t[l] += tgt[l]
        rnd = lambda d, n: math.floor(d / n + 0.5) if d > EPS else 0
        q = {l: rnd(cum_t[l] - cum_a[l], LL) for l in lanes if row_len[l] == LL}
        total = cum_total + P + sum(q.values()) * LL
        for l in [l for l in q if l in prot]:  # a long row raises the total it is measured against
            while cum_a[l] + q[l] * LL < own[l] / 100.0 * total - EPS:
                q[l] += 1
                total += LL
        for l in [l for l in lanes if l in prot and row_len[l] == L]:
            df = own[l] / 100.0 * total - cum_a[l]
            q[l] = max(math.ceil(df / L - EPS) if df > EPS else 0, rnd(cum_t[l] - cum_a[l], L))
        reg_free = [l for l in lanes if l not in q]
        remaining = R - sum(q[l] for l in lanes if l in q and row_len[l] == L)
        if remaining < 0:
            raise ScheduleError(f"step {s}: protected rows exceed the base microbatch")
        dr = {l: (cum_t[l] - cum_a[l]) / L for l in reg_free}
        for l in reg_free:
            q[l] = max(0, math.floor(dr[l] + EPS))
        while sum(q[l] for l in reg_free) > remaining:
            l = min((l for l in reg_free if q[l] > 0), key=lambda l: (dr[l] - q[l], lanes.index(l)))
            q[l] -= 1
        while sum(q[l] for l in reg_free) < remaining:
            l = max(reg_free, key=lambda l: (round(dr[l] - q[l], 9), -lanes.index(l)))
            q[l] += 1
        for l in lanes:
            cum_a[l] += q[l] * row_len[l]
        cum_total = total
        out.append({
            "step": s, "stage": st["name"], "segment": f"{key[0]}{'/fork' if key[1] else ''}",
            "ramp": ({"from": ramp["from"], "to": ramp["to"], "alpha": round(ramp["alpha"], 6)} if ramp else None),
            "mix_pct": {l: round(mix[l], 6) for l in lanes}, "base_window": L, "regular_rows": R,
            "row_len": row_len, "quotas": q, "positions": sum(q[l] * row_len[l] for l in lanes),
            "target_positions": {l: round(tgt[l], 6) for l in lanes},
            "segment_cum_target_positions": {l: round(cum_t[l], 6) for l in lanes},
            "segment_cum_total_positions": cum_total,
            "segment_cum_floor_positions": {l: round(own[l] / 100.0 * cum_total, 6) for l in prot},
            "stage_floor_pct": {l: own[l] for l in prot},
            "segment_cum_quota_positions": dict(cum_a),
            "subpool_mix": _subpool_mix(plan, cfg, s, ramp, st["name"], mix, overrides)})
    return out


def _band_weights(plan, cfg, lane, stage):
    if cfg["lanes"][lane].get("subpools") == "difficulty_band":
        return dict(cfg["a5_difficulty_shares"][stage])
    if cfg["lanes"][lane].get("subpools") == "length_band":
        gates = cfg["a5_reasoning_band_gates"][stage]
        share = plan.get("reasoning_band_sample_share") or {"low": 40, "medium": 35, "high": 20, "ultra": 5}
        return {b: share[b] for b in gates}
    return {"all": 1.0}


def _subpool_mix(plan, cfg, step, ramp, stage, mix, overrides):
    out = {}
    for l in plan["lanes"]:
        if ramp is None:
            w = _band_weights(plan, cfg, l, stage)
        else:  # blend the two stages' band shares by each stage's contribution to this lane
            wp, wn = _band_weights(plan, cfg, l, ramp["from"]), _band_weights(plan, cfg, l, ramp["to"])
            cp, cn = (1 - ramp["alpha"]) * ramp["mix_from"][l], ramp["alpha"] * ramp["mix_to"][l]
            w = {}
            for src, c in ((wp, cp), (wn, cn)):
                t = sum(src.values())
                for b, v in src.items():
                    if t > 0 and c > 0:
                        w[b] = w.get(b, 0.0) + c * v / t
        t = sum(w.values())
        out[l] = {b: v / t for b, v in w.items()} if t > 0 else {}
    return out


# ---------------------------------------------------------------------------------------------------
BAND_ORDER = {"difficulty_band": ["B0_B1", "B2_B3", "B4_B5"], "length_band": ["low", "medium", "high", "ultra"]}


def pool_for(plan, lane, stage, available_pools):
    """Anneal draws from the lane's reserve if it has one (A5 section 7), otherwise from main."""
    if stage == "anneal" and "reserve" in available_pools:
        return "reserve"
    return "main"


def compile_schedule(plan, cfg, store, firewall=None, drill_sources=None, overrides=None, quotas=None):
    lanes = plan["lanes"]
    steps = copy.deepcopy(quotas or compile_quotas(plan, cfg, overrides))
    # ---- sources: every (lane, pool, subpool) list passes the firewall --------------------------------
    sources, blocked = {}, []
    for l in lanes:
        sources[l] = {}
        for pool in ("main", "reserve"):
            for sub in sorted({store.registry[s]["subpool"] for s in store.shards_where(lane=l, pool=pool, split="train")}):
                requested = store.shards_where(lane=l, pool=pool, subpool=sub, split="train")
                if pool == "main" and sub == sorted({store.registry[s]["subpool"] for s in store.shards_where(
                        lane=l, pool="main", split="train")})[0]:
                    for split in (drill_sources or {}).get(l, []):
                        requested = requested + store.shards_where(split=split)  # deliberate misconfiguration
                ok = []
                for sid in requested:
                    v = firewall.shard_violations(sid) if firewall else []
                    if v:
                        blocked.append(firewall.block("mixture_compiler", 0, f"lane {l}/{pool}/{sub} requested {sid}",
                                                      v, shard_id=sid, lane=l, action="source_removed_from_lane"))
                    else:
                        ok.append(sid)
                sources[l].setdefault(pool, {})[sub] = ok
    # ---- pools and sub-pool splits per step ---------------------------------------------------------------
    substitutions = []
    acc = {}
    for st in steps:
        st["pools"], st["subquotas"] = {}, {}
        for l in lanes:
            pool = pool_for(plan, l, st["stage"], sources[l])
            st["pools"][l] = pool
            q = st["quotas"][l]
            mixw = dict(st["subpool_mix"][l])
            have = set(sources[l].get(pool, {}))
            kind = cfg["lanes"][l].get("subpools")
            if kind is None:
                mixw = {"all": 1.0}
            for b in [b for b in mixw if b not in have]:
                order = BAND_ORDER[kind]
                cand = sorted((b2 for b2 in have), key=lambda b2: (abs(order.index(b2) - order.index(b)),
                                                                    order.index(b2)))
                if not cand:
                    raise ScheduleError(f"lane {l}/{pool} has no supply for band {b}")
                mixw[cand[0]] = mixw.get(cand[0], 0.0) + mixw.pop(b)
                if q:
                    substitutions.append({"step": st["step"], "lane": l, "pool": pool, "band": b, "served_by": cand[0]})
            key = (st["segment"], l, pool)
            a = acc.setdefault(key, {"t": {}, "g": {}})
            for b, w in mixw.items():
                a["t"][b] = a["t"].get(b, 0.0) + q * w
            sq = {b: 0 for b in mixw}
            for _ in range(q):  # each row goes to the band with the largest cumulative deficit
                b = max(mixw, key=lambda b: (round(a["t"][b] - a["g"].get(b, 0) - sq[b], 9), -sorted(mixw).index(b)))
                sq[b] += 1
            for b, n in sq.items():
                a["g"][b] = a["g"].get(b, 0) + n
            st["subquotas"][l] = {b: n for b, n in sq.items() if n}
    # ---- supply check -----------------------------------------------------------------------------------
    keep = plan["selector_keep"]
    demand = {}
    for st in steps:
        for l in lanes:
            for b, n in st["subquotas"][l].items():
                f = 1.0 / keep[l] if (l in keep and st["stage"] != "anneal") else 1.0
                k = (l, st["pools"][l], b)
                demand[k] = demand.get(k, 0.0) + n * st["row_len"][l] * f
    scarcity = []
    for (l, pool, b), d in sorted(demand.items()):
        avail = sum(store.registry[s]["token_count"] for s in sources[l][pool][b])
        passes = math.ceil(d / avail) if avail else math.inf
        scarcity.append({"lane": l, "pool": pool, "subpool": b, "candidate_positions_upper_bound": round(d, 1),
                         "available_tokens": avail, "passes_needed": passes, "max_passes": plan["max_passes"],
                         "action": "repeat" if passes > 1 else "none", "satisfiable": passes <= plan["max_passes"]})
        if passes > plan["max_passes"]:
            raise ScheduleError(f"{l}/{pool}/{b} needs {passes} passes > A5 max_passes {plan['max_passes']}")
    sched = {"format": "tdes-schedule/2", "plan_sha256": plan["plan_sha256"], "overrides": overrides,
             "total_steps": plan["total_steps"], "stages": plan["stages"], "protected": plan["protected"],
             "selector_keep": keep, "selector_bypass": plan["selector_bypass"],
             "lanes": {l: {"policy": cfg["lanes"][l]["policy"], "row": cfg["lanes"][l]["row"],
                           "protected": l in plan["protected"], "sources": sources[l]} for l in lanes},
             "steps": steps, "band_substitutions": substitutions, "scarcity": scarcity}
    sched["schedule_sha256"] = sha256_json(sched)
    sched["blocked_sources"] = blocked
    return sched


def lane_demand_positions(quotas, lanes, stage_filter):
    out = {l: 0 for l in lanes}
    for st in quotas:
        if stage_filter(st["stage"]):
            for l in lanes:
                out[l] += st["quotas"][l] * st["row_len"][l]
    return out
