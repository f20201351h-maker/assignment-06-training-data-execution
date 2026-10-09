"""OPUS-style candidate selection with an audit trail, following the A5 contract.

Score (a simplified, deterministic stand-in for OPUS, not the full method). For a candidate row c the score is
cos(G_c, G_p), where G = d(mean loss)/d(output-head weights) = sum_i m_i (softmax_i - onehot(y_i)) h_i^T / n
and G_p is the same quantity for a fixed proxy batch (English held-out text from A2). It asks "would a step on
this row move the head the way the proxy wants?" using only the last layer's gradient. An English proxy is
biased against Indic text, which is the situation the protected floor exists for.

Decide (a pure function of the scores, so the verifier can re-derive every decision). Per A5:
  * selector lanes (A5 `selector_keep` < 1, i.e. web at 0.40): ceil(quota / keep) candidates are scored;
    the best `quota` are ACCEPTED (opus_selected); the next ceil(defer_band_factor * quota) are DEFERRED
    (marginal_utility) and re-scored at the lane's next step by the newer model; the rest are REJECTED
    (low_proxy_utility). A candidate deferred `max_deferrals` times is REJECTED (defer_expired).
  * keep = 1.0 lanes (code, stem): exactly `quota` candidates, all ACCEPTED (keep_1.0_static_filter_only);
    A5 relies on A4's static cleaning there and gives the selector no slack.
  * protected lanes (indic, reasoning, agentic): served by the always-on channel, exactly `quota` rows. They
    are still scored ("shadow score"); a row scoring below this step's web acceptance cutoff - one the
    selector would have thrown away - is ACCEPTED with reason protected_floor_override, otherwise
    always_on_above_cutoff. This is A5's "share == floor" plus the record of what the floor rescued.
  * long_context: selector bypass (A5: a short prefix says little about a long document).
  * anneal stage: selector off for every lane (A5: everything in the anneal was chosen by hand).
"""
import math

import torch
import torch.nn.functional as F


def head_gradients(model, batch):
    """Per-row output-head gradient [N, V*D] for the rows in `batch` (no parameter grads touched)."""
    with torch.no_grad():
        h = model.hidden(batch["tokens"], batch["position_ids"], batch["segment_ids"])
        p = model.head(h).softmax(-1)
        lab = batch["labels"]
        m = (lab >= 0).to(p.dtype)
        idx = lab.clamp(min=0).unsqueeze(-1)
        p.scatter_(-1, idx, p.gather(-1, idx) - 1.0)  # softmax - onehot(label), in place
        p.mul_(m[..., None])
        n = m.sum(1).clamp(min=1.0)
        g = torch.einsum("blv,bld->bvd", p, h) / n[:, None, None]
    return g.reshape(g.shape[0], -1)


def proxy_direction(model, proxy_batch):
    return head_gradients(model, proxy_batch).mean(0, keepdim=True)


def opus_scores(model, cand_batch, gp):
    gc = head_gradients(model, cand_batch)
    sc = F.cosine_similarity(gc, gp.expand_as(gc), dim=1)
    return [float(x) for x in sc], [float(x) for x in gc.norm(dim=1)]


def lane_mode(lane, stage, plan):
    if stage == "anneal":
        return "selector_off"
    if lane in plan["protected"]:
        return "always_on"
    if lane in plan["selector_keep"]:
        return "selector" if plan["selector_keep"][lane] < 1.0 else "keep_all"
    return "bypass"


def n_candidates(lane, quota, stage, plan):
    if quota == 0:
        return 0
    if lane_mode(lane, stage, plan) == "selector":
        return math.ceil(quota / plan["selector_keep"][lane] - 1e-9)
    return quota


def decide(cands, quotas, stage, plan, cfg_opus):
    """cands: dicts with idx, lane, score, deferrals (list order = pool order).
    Returns ({idx: (status, reason)}, info). Pure function of its inputs."""
    out, info = {}, {"cutoff_score": None, "rank_in_lane": {}}
    by_lane = {}
    for c in cands:
        by_lane.setdefault(c["lane"], []).append(c)
    for l, cs in by_lane.items():
        order = sorted(cs, key=lambda c: (-c["score"], c["idx"]))
        for r, c in enumerate(order):
            info["rank_in_lane"][c["idx"]] = r
        if lane_mode(l, stage, plan) == "selector":
            q = quotas[l]
            band = math.ceil(cfg_opus["defer_band_factor"] * q)
            for r, c in enumerate(order):
                if r < q:
                    out[c["idx"]] = ("accepted", "opus_selected")
                elif r < q + band:
                    out[c["idx"]] = (("rejected", "defer_expired") if c["deferrals"] >= cfg_opus["max_deferrals"]
                                     else ("deferred", "marginal_utility"))
                else:
                    out[c["idx"]] = ("rejected", "low_proxy_utility")
            if q:
                info["cutoff_score"] = order[q - 1]["score"]
    for l, cs in by_lane.items():
        mode = lane_mode(l, stage, plan)
        for c in cs:
            if mode == "keep_all":
                out[c["idx"]] = ("accepted", "keep_1.0_static_filter_only")
            elif mode == "always_on":
                cut = info["cutoff_score"]
                out[c["idx"]] = ("accepted", "protected_floor_override" if cut is not None and c["score"] < cut
                                 else "always_on_above_cutoff")
            elif mode == "bypass":
                out[c["idx"]] = ("accepted", "selector_bypass_long_context")
            elif mode == "selector_off":
                out[c["idx"]] = ("accepted", "selector_off_anneal")
    return out, info
