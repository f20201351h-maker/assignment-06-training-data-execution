"""DEVELOPMENT-TIME ONLY. Vendors the small, permitted upstream inputs into inputs/.

    python tools/prepare_inputs.py --a2 ../india-bpe-tokenizer --a4 ../chat-data-cleaning --a5 ../pretraining-data-mixture

The demo (run_demo.py) never runs this script and never reads the upstream folders. It reads only inputs/,
which this script writes once, together with inputs/PROVENANCE.json (source file, source sha256, selection
rule, licence and count for every vendored file). The upstream folders are opened read-only.

What is vendored and why
  a2/tokenizer.json        byte copy of the A2 tokenizer (the frozen tokenizer contract)
  a2/final_config.json     A2's recipe, including the sha256 A2 published for tokenizer.json
  a4/manifest.json         A4 shard manifest (cleaning pipeline hashes, licences, languages)
  a4/sources.json          A4 sources and eval-set revisions
  a5/mixture.yaml          A5 mixture and curriculum contract (compiled at run time, not copied into config)
  corpus/agentic_glaive.jsonl     A4-admitted Glaive conversations (Apache-2.0), multi-step and one-shot
  corpus/indic_anudesh_prompts.jsonl  A4-admitted Anudesh, USER PROMPTS ONLY (human-written, CC-BY-4.0).
                           The Llama-2-generated responses are not vendored: the Llama 2 licence restricts
                           using its outputs to improve other models, and A4 itself did not publish them.
  corpus/wikipedia_en.jsonl       A2's Wikipedia training text (CC BY-SA 4.0): web, STEM, long-context lanes
  corpus/code_stdlib.jsonl        functions from pure-Python CPython stdlib modules (PSF-2.0). Fixture: A4 has no code
  corpus/reasoning_synthetic.jsonl  generated arithmetic traces (CC0). Fixture: A4 has no reasoning traces
  eval/test_items.jsonl    a few items of the A4 decontamination eval sets (GSM8K, HumanEval, MMLU, MATH-500; MIT)
  eval/heldout.jsonl       A2 held-out Wikipedia paragraphs (en/hi/te/ur): validation + OPUS proxy
"""
import argparse
import ast
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "inputs"

STEM_TITLES = ["International Atomic Time", "Agricultural science", "Astronomer", "Arithmetic mean", "Algae",
               "Analysis of variance", "Alkane", "Acid", "Atomic number", "Anatomy", "Ampere", "Algorithm",
               "Asteroid", "Alkali metal", "Amphibian", "Abacus", "Amateur astronomy", "Annual plant"]
# whole short articles, trained in one piece in the long-context lane (A5: complete documents, never cut)
LONG_TITLES = ["Academy Award for Best Production Design", "Actrius", "Animalia (book)", "Transport in Angola",
               "Foreign relations of Angola", "Answer (law)", "Allocution", "Apiales"]
STDLIB_MODULES = ["operator", "urllib.parse", "genericpath", "functools", "gettext", "stat", "shutil", "calendar",
                  "glob", "locale", "base64", "heapq", "posixpath", "tempfile", "uuid", "pprint", "linecache",
                  "tokenize", "colorsys", "bisect", "textwrap", "fnmatch", "statistics", "shlex", "string", "html",
                  "difflib", "quopri"]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stable(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def ntok(tok, s):
    return len(tok.encode(s).ids)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2", required=True)
    ap.add_argument("--a4", required=True)
    ap.add_argument("--a5", required=True)
    a = ap.parse_args(argv)
    A2, A4, A5 = Path(a.a2).resolve(), Path(a.a4).resolve(), Path(a.a5).resolve()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(A2 / "tokenizer" / "tokenizer.json"))
    prov = {"note": "Written by tools/prepare_inputs.py at development time. run_demo.py reads only inputs/.",
            "python": sys.version.split()[0], "files": {}}

    def record(rel, src, rule, licence, count=None, extra=None):
        prov["files"][rel] = {"sha256": sha_file(OUT / rel), "source": src, "selection": rule, "licence": licence,
                              **({"count": count} if count is not None else {}), **(extra or {})}

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    # ---- byte copies of the upstream contracts -------------------------------------------
    for rel, src, label in [("a2/tokenizer.json", A2 / "tokenizer" / "tokenizer.json", "india-bpe-tokenizer/tokenizer/tokenizer.json"),
                            ("a2/final_config.json", A2 / "tokenizer" / "final_config.json",
                             "india-bpe-tokenizer/tokenizer/final_config.json"),
                            ("a4/manifest.json", A4 / "artifacts" / "manifest.json", "chat-data-cleaning/artifacts/manifest.json"),
                            ("a4/sources.json", A4 / "artifacts" / "sources.json", "chat-data-cleaning/artifacts/sources.json"),
                            ("a5/mixture.yaml", A5 / "mixture.yaml", "pretraining-data-mixture/mixture.yaml")]:
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, OUT / rel)
        record(rel, label, "byte copy", "see upstream", extra={"source_sha256": sha_file(src)})

    # ---- A4 Glaive: agentic trajectories --------------------------------------------------
    multi, oneshot = [], []
    with open(A4 / "data" / "clean" / "glaive.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            calls = sum(1 for m in r["messages"] if m.get("tool_call"))
            if calls == 0 or any(m["role"] not in ("system", "user", "assistant", "tool") for m in r["messages"]):
                continue
            body = 0
            for m in r["messages"]:
                c = json.dumps(m["tool_call"], sort_keys=True) if m.get("tool_call") else (m["content"] or "")
                body += ntok(tok, c) + 1
            if body > 1000:  # must fit one 1024-position long row whole (EOS included)
                continue
            row = {"a4_id": r["id"], "a4_src_index": r["src_index"], "a4_shard": "glaive", "tool_calls": calls,
                   "messages": r["messages"], "lang": r["lang"]["doc"]}
            (multi if calls >= 2 else oneshot).append(row)
            if len(multi) >= 36 and len(oneshot) >= 16:
                break
    glaive = multi[:36] + oneshot[:16]
    write_jsonl(OUT / "corpus" / "agentic_glaive.jsonl", glaive)
    record("corpus/agentic_glaive.jsonl", "chat-data-cleaning/data/clean/glaive.jsonl (A4 shard_68191df5e149)",
           "first 36 conversations with >=2 tool calls and first 16 with exactly 1, in A4 file order, whose "
           "rendered length under the A2 tokenizer is <= 1000 tokens", "apache-2.0", len(glaive))

    # ---- A4 Anudesh: user prompts only ------------------------------------------------------
    want = {"hi": 110, "te": 110, "mr": 110, "ur": 20, "hi-Latn": 40, "kn": 6, "ta": 6, "bn": 6}
    got = {k: [] for k in want}
    with open(A4 / "data" / "clean" / "anudesh.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lang = r["lang"]["user"]
            if lang not in want or len(got[lang]) >= want[lang] or r["lang"].get("user_status") != "ok":
                continue
            u = [m["content"] for m in r["messages"] if m["role"] == "user"][0].strip()
            if not u or len(u) > 600:
                continue
            got[lang].append({"a4_id": r["id"], "a4_src_index": r["src_index"], "a4_shard": "anudesh",
                              "language": lang.split("-")[0], "script_hint": "Latn" if lang.endswith("Latn") else None,
                              "text": u})
            if all(len(got[k]) >= want[k] for k in want):
                break
    prompts = [x for k in want for x in got[k]]
    write_jsonl(OUT / "corpus" / "indic_anudesh_prompts.jsonl", prompts)
    record("corpus/indic_anudesh_prompts.jsonl", "chat-data-cleaning/data/clean/anudesh.jsonl (A4 shard_6aefac17acda)",
           "first user prompt of the first N conversations per A4 user-language tag (" +
           ", ".join(f"{k}:{len(got[k])}" for k in want) + "); responses NOT copied (Llama-2 output licence). "
           "kn/ta/bn are included on purpose: A4 admitted them but the A2 tokenizer cannot encode their scripts",
           "cc-by-4.0 (AI4Bharat indic-align; attribution required)", len(prompts))

    # ---- A2 Wikipedia: web, STEM, long-context ----------------------------------------------
    with gzip.open(A2 / "data" / "raw" / "external" / "en.train.jsonl.gz", "rt", encoding="utf-8") as f:
        arts = [json.loads(l) for l in f]
    wiki = []
    for r in arts:
        title = r["title"]
        if title in LONG_TITLES:
            lane, cap = "long_context", None
        elif title in STEM_TITLES:
            lane, cap = "stem", 700
        elif "(disambiguation)" in title or title.startswith("List of") or ntok(tok, r["text"]) < 1200:
            continue
        else:
            lane, cap = "web", 1500
        paras = [p.strip() for p in r["text"].split("\n") if p.strip()]
        if cap is None:
            text = "\n".join(paras)
        else:  # leading whole paragraphs up to ~cap tokens (an excerpt, never a cut paragraph)
            keep, n = [], 0
            for p in paras:
                k = ntok(tok, p)
                if keep and n + k > cap:
                    break
                keep.append(p)
                n += k
            text = "\n".join(keep)
        wiki.append({"wiki_id": r["id"], "title": title, "lane_hint": lane, "text": text})
    web = [w for w in wiki if w["lane_hint"] == "web"]
    web = sorted(web, key=lambda w: stable("web", w["wiki_id"]))[:44]
    wiki = sorted([w for w in wiki if w["lane_hint"] != "web"] + web, key=lambda w: int(w["wiki_id"]))
    write_jsonl(OUT / "corpus" / "wikipedia_en.jsonl", wiki)
    record("corpus/wikipedia_en.jsonl", "india-bpe-tokenizer/data/raw/external/en.train.jsonl.gz "
           "(HF wikimedia/wikipedia 20231101.en, as fetched by A2)",
           "web: 44 general articles (>=1200 tokens, not lists/disambiguation; chosen by stable hash), leading whole "
           "paragraphs up to ~1500 tokens; stem: listed science/math titles, up to ~700 tokens; long_context: listed "
           "short articles, complete", "cc-by-sa-4.0", len(wiki),
           extra={"source_sha256": sha_file(A2 / "data" / "raw" / "external" / "en.train.jsonl.gz")})

    # ---- held-out (validation + OPUS proxy) from A2 held-out text ---------------------------
    held = []
    for lang, nval in [("en", 3), ("hi", 3), ("te", 3), ("ur", 3)]:
        with gzip.open(A2 / "data" / "raw" / "external" / f"{lang}.heldout.jsonl.gz", "rt", encoding="utf-8") as f:
            docs = [json.loads(l) for l in f]
        paras = []
        for d in docs:
            for p in d["text"].split("\n"):
                p = p.strip()
                if len(p.split()) >= 40 and ntok(tok, p) <= 250 and tok.encode(p).ids.count(0) == 0:
                    paras.append((d["id"], d["title"], p))
        val = paras[:nval]
        for wid, title, p in val:
            held.append({"split": "validation", "language": lang, "wiki_id": wid, "title": title, "text": p})
        if lang == "en":
            used = {wid for wid, _, _ in val}
            prox = [x for x in paras if x[0] not in used][:4]
            for wid, title, p in prox:
                held.append({"split": "proxy", "language": lang, "wiki_id": wid, "title": title, "text": p})
    write_jsonl(OUT / "eval" / "heldout.jsonl", held)
    record("eval/heldout.jsonl", "india-bpe-tokenizer/data/raw/external/{en,hi,te,ur}.heldout.jsonl.gz (A2 held-out text)",
           "first 3 paragraphs per language with >=40 words, <=250 A2 tokens and no <unk> -> validation; 4 more "
           "English paragraphs from other articles -> OPUS proxy", "cc-by-sa-4.0", len(held))

    # ---- A4 eval sets: registered test items ----------------------------------------------
    import pandas as pd
    tests = []
    e = A4 / "data" / "eval"
    g = pd.read_parquet(e / "gsm8k_test.parquet")
    for i in (0, 1, 2, 3, 4, 5):
        tests.append({"benchmark_id": "gsm8k", "item_id": f"test-{i}", "text": g.iloc[i]["question"] + "\n" +
                      g.iloc[i]["answer"]})
    h = pd.read_parquet(e / "humaneval_test.parquet")
    for i in (0, 1, 2):
        tests.append({"benchmark_id": "humaneval", "item_id": h.iloc[i]["task_id"],
                      "text": h.iloc[i]["prompt"] + h.iloc[i]["canonical_solution"]})
    m = pd.read_parquet(e / "mmlu_test.parquet")
    for i in (0, 1, 2, 3):
        r = m.iloc[i]
        tests.append({"benchmark_id": "mmlu", "item_id": f"{r['subject']}-{i}",
                      "text": r["question"] + "\n" + "\n".join(str(c) for c in r["choices"])})
    with open(e / "math500_test.jsonl", encoding="utf-8") as f:
        for i, line in zip(range(3), f):
            r = json.loads(line)
            tests.append({"benchmark_id": "math500", "item_id": r.get("unique_id", f"test-{i}"),
                          "text": r["problem"] + "\n" + r["solution"]})
    srcs = json.loads((A4 / "artifacts" / "sources.json").read_text(encoding="utf-8"))["eval_sets"]
    rev = {s["name"]: s["revision"] for s in srcs}
    for t in tests:
        t["version"] = rev[t["benchmark_id"]]
    write_jsonl(OUT / "eval" / "test_items.jsonl", tests)
    record("eval/test_items.jsonl", "chat-data-cleaning/data/eval/* (the eval sets A4 decontaminated against)",
           "gsm8k test 0-5, humaneval 0-2, mmlu test 0-3, math500 first 3", "mit", len(tests))

    # ---- code fixture: pure-Python stdlib functions -----------------------------------------
    import importlib
    import inspect
    code = []
    for mod in STDLIB_MODULES:
        mm = importlib.import_module(mod)
        src_path = Path(inspect.getsourcefile(mm))
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        lines = src_path.read_text(encoding="utf-8").splitlines()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("__"):
                body = "\n".join(lines[node.lineno - 1 - len(node.decorator_list):node.end_lineno])
                n = ntok(tok, body)
                if 12 <= n <= 1000:
                    code.append({"module": mod, "function": node.name, "file": src_path.name, "text": body,
                                 "file_sha256": sha_file(src_path)})
    write_jsonl(OUT / "corpus" / "code_stdlib.jsonl", code)
    record("corpus/code_stdlib.jsonl", f"CPython {sys.version.split()[0]} Lib/ ({', '.join(STDLIB_MODULES)})",
           "every top-level public function with 12..1000 A2 tokens", "psf-2.0", len(code))

    # ---- reasoning fixture: generated traces, two length bands -------------------------------
    reas = []
    for i in range(70):
        h1 = int(stable("rs-low", i)[:8], 16)
        a, b = 11 + h1 % 89, 3 + (h1 >> 8) % 17
        q, r_ = divmod(a, b)
        reas.append({"band_hint": "low", "text": f"Question: what is {a} divided by {b}?\nTrace: {b} x {q} = "
                     f"{b * q}, and {a} - {b * q} = {r_} is left over.\nAnswer: {q} remainder {r_}."})
    i = 0
    while sum(1 for x in reas if x["band_hint"] == "medium") < 70:
        h2 = int(stable("rs-med", i)[:12], 16)
        i += 1
        n, p, k, c, r_ = 20 + h2 % 60, 2 + (h2 >> 8) % 7, 3 + (h2 >> 16) % 5, 5 + (h2 >> 24) % 20, (h2 >> 32) % 90
        nb, pc = n * p, k * c
        budget = nb + pc + r_
        text = (f"Question: {n} notebooks cost {p} rupees each and {k} pens cost {c} rupees each. The budget is "
                f"{budget} rupees. How much is left?\nTrace: notebooks {n} x {p} = {nb}. Pens {k} x {c} = {pc}. "
                f"Spent {nb} + {pc} = {nb + pc}. Left {budget} - {nb + pc} = {r_}. Check {r_} + {nb + pc} = "
                f"{budget}.\nAnswer: {r_} rupees.")
        if 64 <= ntok(tok, text) <= 120:  # A5 "medium" band starts at 64; must fit a 128-position row with EOS
            reas.append({"band_hint": "medium", "text": text})
    write_jsonl(OUT / "corpus" / "reasoning_synthetic.jsonl", reas)
    record("corpus/reasoning_synthetic.jsonl", "tools/prepare_inputs.py generator (hash-seeded)",
           "70 short division traces + 70 multi-step budget traces", "cc0-1.0 (self-generated fixture)", len(reas))

    (OUT / "PROVENANCE.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False, sort_keys=True),
                                         encoding="utf-8", newline="\n")
    for rel, v in sorted(prov["files"].items()):
        print(f"{rel:40s} {v.get('count', ''):>5} {v['sha256'][:12]}  {v['licence']}")


if __name__ == "__main__":
    main(sys.argv[1:])
