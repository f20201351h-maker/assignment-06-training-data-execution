"""Vendored inputs -> documents -> A6 ingress gate -> admitted documents with lane / pool / sub-pool.

Sources (see inputs/PROVENANCE.json):
  a4:glaive    A4-admitted function-calling conversations      -> agentic (>=2 tool calls: main pool,
                                                                  1 tool call: anneal reserve, as A5 reserves
                                                                  the whole one-shot pool)
  a4:anudesh   A4-admitted Anudesh, human-written user prompts  -> indic (A5 tier D; difficulty sub-pools)
  a2:wikipedia A2's Wikipedia training text                     -> web, stem, long_context (NOT A4-admitted)
  fixture:stdlib, fixture:reasoning                             -> code / long_context, reasoning (fixtures)
Eval-side: registered test items (A4's decontamination sets), validation and OPUS proxy (A2 held-out text).

The A6 ingress gate runs on every training document, A4-admitted or not, in this order:
  1. licence must be in the allow-list;
  2. A2 tokenizer coverage: <unk> rate <= max_unk_rate (A4 admitted 15 languages, A2 encodes 4 scripts);
  3. exact duplicate after A2 normalisation -> dropped;
  4. PII (e-mail, phone) -> redacted. Public library source code (fixture:stdlib) is exempt and marked so: the
     regexes misfire on digit literals such as "0123456789" and on mailing-list message ids in comments;
  5. contamination: any shared word n-gram with a registered eval/validation/proxy item, or a canary -> dropped.
Every decision is written to manifests/admission_report.json.
"""
import json
import re
import unicodedata
from pathlib import Path

from .hashing import sha256_json, sha256_text_file, stable_u64

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[ -]?)?\d{5}[ -]?\d{5}(?!\d)")
CANARY = "TDES-EVAL-CANARY-7f3a9c"   # registered in eval_registry.json; training text containing it is rejected
PII_EXEMPT_SOURCES = {"fixture:stdlib"}
SCRIPTS = {"Deva": (0x0900, 0x097F), "Beng": (0x0980, 0x09FF), "Gujr": (0x0A80, 0x0AFF), "Guru": (0x0A00, 0x0A7F),
           "Orya": (0x0B00, 0x0B7F), "Taml": (0x0B80, 0x0BFF), "Telu": (0x0C00, 0x0C7F), "Knda": (0x0C80, 0x0CFF),
           "Mlym": (0x0D00, 0x0D7F), "Arab": (0x0600, 0x06FF)}


def script_of(text):
    counts = {}
    for ch in text:
        o = ord(ch)
        sc = next((k for k, (a, b) in SCRIPTS.items() if a <= o <= b), "Latn" if ch.isascii() and ch.isalpha() else None)
        if sc:
            counts[sc] = counts.get(sc, 0) + 1
    return max(sorted(counts), key=lambda k: counts[k]) if counts else "Zyyy"


def doc_text(doc):
    if "messages" in doc:
        return "\n".join((json.dumps(m["tool_call"], sort_keys=True) if m.get("tool_call") else (m["content"] or ""))
                         for m in doc["messages"])
    return doc["text"]


def fingerprint_words(text):
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


def shingle_hashes(text, n):
    import hashlib
    w = fingerprint_words(text)
    return {hashlib.sha256(" ".join(w[i:i + n]).encode()).hexdigest()[:16] for i in range(len(w) - n + 1)}


def eval_fingerprints(text, n):
    """(n, hashes) for one eval item: n-gram shingles, or the whole word sequence if it is shorter than n."""
    w = fingerprint_words(text)
    k = min(n, len(w))
    return k, shingle_hashes(text, k)


def jl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def load_inputs(repo, cfg):
    """Raw documents (train candidates) and eval-side documents, all with explicit provenance."""
    inp = Path(repo) / cfg["inputs_dir"]
    prov = json.loads((inp / "PROVENANCE.json").read_text(encoding="utf-8"))
    docs = []
    for r in jl(inp / "corpus" / "agentic_glaive.jsonl"):
        docs.append({"doc_id": f"glaive-{r['a4_id']}", "source": "a4:glaive", "source_id": "a4/glaive",
                     "license": "apache-2.0", "language": "en", "messages": r["messages"], "tool_calls": r["tool_calls"],
                     "a4_admitted": True, "a4_record_id": r["a4_id"], "lane_hint": "agentic"})
    for r in jl(inp / "corpus" / "indic_anudesh_prompts.jsonl"):
        docs.append({"doc_id": f"anudesh-{r['a4_id']}", "source": "a4:anudesh", "source_id": "a4/anudesh-user-prompts",
                     "license": "cc-by-4.0", "language": r["language"], "text": r["text"], "a4_admitted": True,
                     "a4_record_id": r["a4_id"], "lane_hint": "indic"})
    for r in jl(inp / "corpus" / "wikipedia_en.jsonl"):
        docs.append({"doc_id": f"wiki-{r['wiki_id']}", "source": "a2:wikipedia", "source_id": "a2/wikimedia-20231101-en",
                     "license": "cc-by-sa-4.0", "language": "en", "text": r["text"], "title": r["title"],
                     "a4_admitted": False, "lane_hint": r["lane_hint"]})
    for r in jl(inp / "corpus" / "code_stdlib.jsonl"):
        docs.append({"doc_id": f"py-{r['module']}.{r['function']}", "source": "fixture:stdlib",
                     "source_id": f"cpython-lib/{r['file']}", "license": "psf-2.0", "language": "python",
                     "text": r["text"], "a4_admitted": False, "lane_hint": "code"})
    for i, r in enumerate(jl(inp / "corpus" / "reasoning_synthetic.jsonl")):
        docs.append({"doc_id": f"rs-{r['band_hint']}-{i:03d}", "source": "fixture:reasoning",
                     "source_id": "a6-generator/reasoning-v1", "license": "cc0-1.0", "language": "en",
                     "text": r["text"], "a4_admitted": False, "lane_hint": "reasoning"})
    evals = []
    for i, r in enumerate(jl(inp / "eval" / "test_items.jsonl")):
        evals.append({"doc_id": f"test-{r['benchmark_id']}-{r['item_id']}".replace("/", "_"), "split": "test",
                      "benchmark_id": r["benchmark_id"], "item_id": r["item_id"], "version": r["version"],
                      "text": r["text"], "language": "en", "license": "mit", "source_id": f"a4-eval/{r['benchmark_id']}"})
    for i, r in enumerate(jl(inp / "eval" / "heldout.jsonl")):
        evals.append({"doc_id": f"{r['split']}-{r['language']}-{i:02d}", "split": r["split"], "language": r["language"],
                      "text": r["text"], "license": "cc-by-sa-4.0", "source_id": f"a2-heldout/{r['language']}",
                      "benchmark_id": "a2-heldout" if r["split"] == "validation" else None, "item_id": None,
                      "version": "wikimedia-20231101"})
    return docs, evals, prov


def ingress_hash(cfg):
    return sha256_json({"gate": "tdes-a6-ingress/1", "cleaning": cfg["cleaning"],
                        "corpus_py_sha256": sha256_text_file(Path(__file__))})


def _redact(s):
    s, a = EMAIL_RE.subn("<EMAIL>", s)
    s, b = PHONE_RE.subn("<PHONE>", s)
    return s, a + b


def admit(docs, evals, tok, cfg, drill=None):
    """Returns admitted docs (deep copies, lane/subpool assigned), the eval docs, and the decision report."""
    c = cfg["cleaning"]
    n = c["fingerprint_ngram_words"]
    allowed = set(c["allowed_licenses"])
    efp = {}
    for e in evals:
        k, hs = eval_fingerprints(e["text"], n)
        efp.setdefault(k, set()).update(hs)
    docs = [json.loads(json.dumps(d)) for d in docs]
    if drill:  # a web document that quietly embeds a registered test item (copy-paste contamination)
        item = next(e for e in evals if e.get("benchmark_id") == drill["benchmark_id"] and e["item_id"] == drill["item_id"])
        host = next(d for d in sorted(docs, key=lambda d: d["doc_id"]) if d["lane_hint"] == drill["host_lane"])
        docs.append(dict(host, doc_id=host["doc_id"] + "-DRILL-contaminated", text=host["text"][:400] + "\n" + item["text"],
                         drill="ingress_contaminated_document"))
    report, admitted, seen = [], [], {}
    for d in sorted(docs, key=lambda d: d["doc_id"]):
        reasons, pii = [], 0
        exempt = d["source"] in PII_EXEMPT_SOURCES
        if d["license"] not in allowed:
            reasons.append(f"license_not_allowed:{d['license']}")
        if exempt:
            pass
        elif "messages" in d:
            for m in d["messages"]:
                if m.get("content"):
                    m["content"], k = _redact(m["content"])
                    pii += k
        else:
            d["text"], pii = _redact(d["text"])
        text = doc_text(d)
        unk = tok.unk_rate(text)
        if unk > c["max_unk_rate"]:
            reasons.append(f"a2_tokenizer_coverage:unk_rate={unk:.3f}")
        key = sha256_json([tok.normalize(m.get("content") or json.dumps(m.get("tool_call"), sort_keys=True))
                           for m in d["messages"]] if "messages" in d else tok.normalize(d["text"]))
        if key in seen:
            reasons.append(f"exact_duplicate_of:{seen[key]}")
        hits = sum(len(shingle_hashes(text, k) & hs) for k, hs in efp.items())
        if hits:
            reasons.append(f"eval_contamination:{hits}_shared_word_ngrams")
        if CANARY in text:
            reasons.append("eval_canary_present")
        d["script"] = script_of(text)
        report.append({"doc_id": d["doc_id"], "source": d["source"], "lane_hint": d["lane_hint"], "language": d["language"],
                       "script": d["script"], "a4_admitted": d["a4_admitted"], "unk_rate": round(unk, 5),
                       "pii_redactions": pii, "decision": "admitted" if not reasons else "dropped", "reasons": reasons,
                       **({"drill": d["drill"]} if d.get("drill") else {})})
        if not reasons:
            seen[key] = d["doc_id"]
            d["pii_status"] = "exempt_public_source_code" if exempt else ("redacted" if pii else "clean")
            admitted.append(d)
    for e in evals:
        report.append({"doc_id": e["doc_id"], "source": e["source_id"], "lane_hint": e["split"], "language": e["language"],
                       "decision": f"registered_{e['split']}_only", "reasons": ["never_train"]})
    return admitted, evals, report


def assign_lanes(admitted, tok, cfg, plan, anneal_demand):
    """Lane, pool and sub-pool of every admitted document (deterministic)."""
    LL = plan["long_row_len"]
    base_min = min(s["seq_len"] for s in plan["stages"])
    gates = cfg["a5_reasoning_band_gates"]["band_max_tokens"]
    dp = cfg["indic_difficulty_proxy"]
    for d in admitted:
        n = len(tok.render_document(d)[0])
        d["n_tokens"] = n
        h = d["lane_hint"]
        d["pool"], d["subpool"], d["a5_tier"] = "main", "all", None
        if h == "code":
            if n <= base_min:
                d["lane"] = "code"
            elif n <= LL:
                d["lane"], d["a5_tier"] = "long_context", "repo_or_long_file"
            else:
                d["lane"] = None
        elif h == "agentic":
            d["lane"] = "agentic" if n <= LL else None
            d["pool"] = "main" if d["tool_calls"] >= 2 else "reserve"
            d["a5_tier"] = "agentic_multistep" if d["tool_calls"] >= 2 else "agentic_oneshot"
        elif h == "indic":
            d["lane"], d["a5_tier"] = "indic", "D_synthetic"
            d["subpool"] = "B0_B1" if n <= dp["B0_B1_max_tokens"] else ("B2_B3" if n <= dp["B2_B3_max_tokens"] else "B4_B5")
        elif h == "reasoning":
            d["lane"] = "reasoning" if n <= base_min else None
            d["subpool"] = next(b for b in ("low", "medium", "high", "ultra") if n <= gates[b] + 1)
        elif h == "long_context":
            d["lane"], d["a5_tier"] = ("long_context", "complete_document") if n <= LL else (None, None)
        else:
            d["lane"] = h
    # anneal reserves: A5 reserve fraction of the lane's documents (stable-hash order), raised if the demo's
    # anneal demand would otherwise need more than A5's max_passes over the reserve
    notes = {}
    for lane, frac in plan["reserve_fraction"].items():
        pool_docs = sorted((d for d in admitted if d.get("lane") == lane and d["pool"] == "main"),
                           key=lambda d: (stable_u64(cfg["seed"], "reserve", lane, d["doc_id"]), d["doc_id"]))
        need = anneal_demand.get(lane, 0) / plan["max_passes"]
        k_frac = max(1, -(-int(frac * len(pool_docs) * 1e6) // int(1e6)))
        k, tot = 0, 0
        while k < len(pool_docs) and (k < k_frac or tot < need):
            tot += pool_docs[k]["n_tokens"]
            k += 1
        for d in pool_docs[:k]:
            d["pool"] = "reserve"
        notes[lane] = {"a5_reserve_fraction": frac, "docs_in_lane": len(pool_docs), "reserve_docs_by_fraction": k_frac,
                       "reserve_docs_used": k, "reserve_tokens": tot, "anneal_demand_positions": anneal_demand.get(lane, 0),
                       "raised_to_respect_max_passes": k > k_frac}
    dropped = [d["doc_id"] for d in admitted if d.get("lane") is None]
    return [d for d in admitted if d.get("lane")], dropped, notes
