# Training data execution system

Built as ERA V5 Session 6 assignment.

This repository takes three upstream pieces of my earlier work and turns them into the stream a training loop
actually consumes:

- **A2**, a 10K-token BPE tokenizer for English, Hindi, Telugu and Urdu (`india-bpe-tokenizer`);
- **A4**, two cleaned chat datasets, Glaive function calling and IndicAlign Anudesh (`chat-data-cleaning`);
- **A5**, a pretraining mixture and curriculum specification (`pretraining-data-mixture`).

It then proves, from files on disk, what was consumed, why, what the model measured on it, and that the stream
can be rebuilt after a crash. The short labels A2, A4 and A5 are used throughout the code and the artifacts.

```bash
pip install -r requirements.txt
python run_demo.py          # rebuilds artifacts/ (about 3 to 6 minutes on a laptop CPU)
python -m pytest            # 37 tests, about 5 minutes
```

`run_demo.py` exits 0 only if the verifier marks every requirement PASS. No GPU, no download, and nothing outside
this folder is read at run time: the small upstream inputs are vendored in `inputs/` with their hashes and
licences in `inputs/PROVENANCE.json`. They are byte copies taken when the inputs were vendored, so their hashes are stable even though the
upstream repositories later received comment- and wording-only edits.

## Where the pieces came from

The execution plumbing is one process per phase, a real `os._exit(137)` crash in the middle of a ledger write, the
hash-chained ledger and its recovery, checkpoint read-back, replay and fork. The data side is built on the upstream
work: the A2 tokenizer (rather than a small tokenizer trained on template text), the A5 lanes (rather than invented
ones), and OPUS decisions that do not move slots between lanes, which would leave web text above its planned share.

| Upstream | What I used | What did not fit, and what I did |
|---|---|---|
| A2 tokenizer | `tokenizer.json` byte for byte (sha256 `45aa8369f867472f`, the hash A2 published) | A2 has no EOS, PAD or role tokens. An adapter appends 7 control tokens after the 10,000 A2 ids; text can never produce them. A2 also collapses all whitespace, so code loses newlines. I left that alone (changing it would make a different tokenizer) and flagged it |
| A4 data | Glaive conversations (Apache-2.0) for the agentic lane, Anudesh prompts (CC-BY-4.0) for Indic | Anudesh answers were written by Llama-2, whose licence restricts training other models, so only the human prompts are vendored. A4 admitted 15 languages but A2 encodes 4 scripts: the ingress gate dropped 38 Marathi, 6 Kannada, 6 Bengali and 5 Tamil prompts for `<unk>` rate above 1% (and 36 stdlib functions, mostly for backticks A2 never saw) |
| A4 supply vs A5 lanes | | A4 cleaned only two SFT shards. Web, STEM and long-context come from A2's Wikipedia text, code from CPython stdlib functions, reasoning from a small generated set. Their manifests say `a4_admitted: false` |
| A5 mixture | `mixture.yaml` itself, compiled at run time | Indic tier A/B/C supply is zero (Anudesh is tier D), and A5's agentic Tier A does not exist yet. The lanes keep their protected shares and the gaps are reported in `upstream_contracts.json` |

## The path

```
inputs/ (A2 tokenizer, A4 records, A5 yaml)
  -> ingress gate -> 52 immutable shards + manifests -> eval registry
  -> A5 schedule compiled to per-step lane quotas -> candidate rows packed per data type
  -> eval firewall -> OPUS decisions -> microbatches -> tiny CPU model (real cross-entropy)
  -> consumption ledger + learning ledger -> checkpoint (with ledger offset and next batch)
  -> crash -> resume in a new process -> replay -> fork -> audit -> performance -> verify
```

The model is a 2-layer transformer with d=64 and the A2 vocabulary. It exists so the learning ledger holds real
per-token losses. Training it is not the point.

## Shards and the frozen tokenizer

A shard's id is a hash of its token bytes, its loss-eligibility bytes, its document index, the tokenizer id and
the ingress transform. The same inputs give the same ids: the build makes every shard twice from scratch and
compares (`registry a4d475fe5b1a582a` both times), and a separate clean copy of the repo rebuilt all 52 shards
byte for byte. The store recomputes the hash on every open. The demo flips one byte in a copy of a shard, and the
store rejects it. It also swaps the EOS and PAD ids in a variant tokenizer, and every shard is refused under it.

## Packing by data type

Rows have a fixed length. A5's windows are divided by 32: 128 positions in S1 and S2, 256 from S3, and a separate
1024-position microbatch for long-context and agentic rows. Agentic trajectories in Glaive run 430 to 900 A2 tokens,
too long for the base window, which A5 also notes at full scale.

| Lane | Policy | Utilization | Loss-bearing | Pad-only, same docs |
|---|---|---|---|---|
| web, stem, indic | concat and chop, EOS between documents | 1.000 | 0.986 to 0.989 | 0.54 to 0.71 |
| code | whole functions, best fit over 8 | 0.945 (128), 0.961 (256) | 0.93 to 0.95 | 0.26 to 0.54 |
| reasoning | whole traces, never cut | 0.74 to 0.92 | 0.73 to 0.91 | 0.35 to 0.37 |
| agentic | whole trajectories, loss only on assistant and tool-call tokens | 0.861 | 0.268 | 0.646 |
| long_context | whole documents, best fit | 0.892 | 0.889 | 0.324 |

For every row the same rules hold. Positions restart at 0 for each segment. There is no loss across a document
boundary or on padding. Attention is causal and stays inside the segment. The verifier rebuilds all 487 consumed
rows with its own loop-based packer and gets the same hashes. It also checks that no user, system or tool token in
an agentic row carries loss. The agentic row's 0.268 loss-bearing fraction is that rule at work.

## The A5 mixture, executed

The compiler turns each A5 stage into steps (40 main steps split 8/12/12/8, plus 2 anneal steps), blends mixes
linearly across each boundary, and moves the 256 window two steps before S3 starts, because A5 says the window and
the mixture should not change at the same step. Protected lanes (indic, reasoning, agentic) round up to whole rows
against the positions actually consumed, so their stage-cumulative share never drops below the A5 floor after any
step. Web, code and stem absorb the rounding.

| Stage | web | code | stem | indic | reasoning | agentic | long_context |
|---|---|---|---|---|---|---|---|
| S1 (A5 / actual %) | 70 / 69.5 | 14 / 14.1 | 6 / 6.2 | 10 / 10.2 | 0 / 0 | 0 / 0 | 0 / 0 |
| S2 | 55.5 / 55.7 | 26 / 25.5 | 8 / 7.8 | 10 / 10.4 | 0.5 / 0.5 | 0 / 0 | 0 / 0 |
| S3 | 42.5 / 41.0 | 34 / 33.0 | 12 / 11.0 | 10 / 10.0 | 1 / 1.0 | 0.5 / 4.0 | 0 / 0 |
| S4 | 31.5 / 31.2 | 28 / 27.5 | 9 / 8.8 | 10 / 10.0 | 2 / 2.5 | 1.5 / 5.0 | 18 / 15.0 |
| anneal | 22 / 16.7 | 23 / 16.7 | 12 / 4.2 | 20 / 20.8 | 8 / 8.3 | 5 / 16.7 | 10 / 16.7 |

The large gaps are honest rounding. One agentic trajectory row is 1024 positions, which is 4% of S3. The anneal is
2 steps (A5's 3% would be 1.25), so one long row there is a sixth of the stage. The verifier checks every step
against targets it recomputes from `mixture.yaml` and requires each deviation to stay within whole-row rounding.

Inside the Indic lane the A5 difficulty ladder runs on a token-length proxy (B0-B1 up to 20 tokens, B4-B5 above 45),
so S1 draws mostly short prompts and the anneal mostly long ones. That is an easier-first Indic curriculum. Reasoning uses only the bands A5 allows per stage. A5 asks for high-band traces in S3/S4 and the
anneal, and none exist, so the nearest band (medium) serves them. All 7 substitutions are listed in the schedule.

## OPUS

The score is a simplified, deterministic stand-in for OPUS: the cosine between a candidate row's output-head
gradient and the gradient of a fixed English proxy batch (A2 held-out text). It is not the full method. The
decision rule follows A5. Only web is selected, at keep 0.40: 28 candidates for 11 slots in S1, with the next band
deferred and re-scored by the next model. Code and STEM are kept at 1.0. Protected lanes are always on, but they are
still scored, so the trail shows what the floor rescued. Long-context bypasses the selector, and the anneal runs
with it off.

| Outcome | Count | Reasons |
|---|---|---|
| accepted | 487 | 253 selected, 158 keep-1.0, 55 protected-floor override, 18 anneal, 3 long-context bypass |
| deferred | 126 | marginal utility |
| rejected | 277 | 253 low proxy utility, 16 deferred too often, 6 stage changed under them, 2 eval firewall |

All 55 protected rows outside the anneal (49 Indic, 4 reasoning, 2 agentic) scored below that step's web cutoff,
so every one was a rescue. Their mean token loss was 8.17 against 7.21 for everything else. The English proxy
undervalues exactly the data the model knows least, which is the case for having a floor at all. The web selector
accepted 39% of what it scored.

## Eval and validation firewall

Test items are real ones from the sets A4 decontaminated against (6 GSM8K, 3 HumanEval, 4 MMLU, 3 MATH-500). The
validation and proxy text is A2's held-out Wikipedia. The demo attacks four gates:

- **Ingress.** A web document with GSM8K test-0 pasted in is dropped (71 shared 13-word n-grams, A4's n).
- **Mixture compiler.** It is handed a web source list containing the test and validation shards, and removes both.
- **Candidate gate.** At step 3 the injected test and validation rows are rejected before scoring.
- **Batch gate.** At step 5 a validation row placed straight into the loss-bearing batch makes the step refuse to run.

Afterwards the verifier rebuilds fingerprints from the raw eval files and scans all 2,529 consumed spans of the
main, reference and fork runs: zero hits. Validation loss is read with gradients off, and every read is logged.

## Ledgers and the learning signal

Each consumption record names the run, branch, step, checkpoint, microbatches, packed samples, shard spans,
mask/position/attention hashes, lane, stage, tokenizer id and OPUS decision, and chains to the previous record.
The learning ledger has one line per loss-bearing token (88,192 of them), each with its token id, shard, document
offset, loss and perplexity, plus 1,126 sample lines with loss before and after the update, the hardest tokens and
the checkpoint around them. Each consumption record stores the hash of its step's token lines, so the two cannot
drift apart. Step loss fell from 9.21 (uniform over 10,007 ids) to 6.75. English validation loss fell from 8.16 to
6.99. Hindi and Telugu fell by about half as much (8.96 to 8.42, 9.24 to 8.79), and Urdu barely moved (9.48 to
9.45). A4's Anudesh shard has only 2 usable Urdu prompts, so that is what I would expect, and it is the kind of
gap the ledger is meant to expose.

## Crash, resume, replay, fork

- **Crash.** At step 31 the training process writes half of that step's ledger record (6,260 bytes) and calls
  `os._exit(137)`. The last checkpoint is `main@s00028`. Steps 29 and 30 were already committed, but no checkpoint
  covers them.
- **Resume.** A new process trims the ledger back to offset 28, after checking that the record there has the head
  hash the checkpoint remembers. It keeps the torn file in `crash_forensics/` and restores the model, optimizer,
  LR schedule and data position. The first resumed batch is `bat-s00029-38e71de9eec7` (hash `f4755740a8d70844`,
  8 samples). It is equal, id by id, span by span and row hash by row hash, to the next batch the checkpoint
  declared and to the record the dead process had written. The checkpoint's declaration comes from the same
  planner, so the two independent proofs are that pre-crash record and an uninterrupted reference run. The reference
  run matches all 42 steps, every batch hash and the model hash after every step, so nothing was skipped or repeated.
- **Replay.** Steps 8 to 14 are rebuilt from the ledger's spans, not re-planned, and retrained from `main@s00007`.
  The ledger is replayed, not the planning code. Batch ids, spans, row hashes, per-token losses
  and the final model all match. The stream fingerprint is `1c168a1bbdbde860` for both, and an independent re-plan
  agrees too.
- **Fork.** `fork-s14-a5-h1-indic16` starts from `main@s00014` and runs A5's proxy hypothesis H1: the 16% Indic arm.
  Its ledger chains from the parent's head at offset 14. From step 15 every batch differs from main
  (`bat-s00015-8ccd2f03ddc2` vs `bat-s00015-2a370230bcd8`), with Indic at 17.7% in S2 and 20.0% in S3.

## Performance (CPU, one thread)

Packing utilization is 0.978 and 0.946 of all positions carry loss (padding 0.022, context-only 0.032). The
committed run delivered 2,025 useful loss-bearing tokens/s (1,766 end to end, counting checkpoints and the two
steps lost in the crash) against 2,139 raw positions/s. Resume took 3.1 s and the 7-step replay 10.0 s. A cold
shard open with hash check takes about 1 ms at the median. GPU idle time is reported as null because there is no GPU. These timings move with
machine load.

## Evidence and tests

`artifacts/evidence.md` and `evidence.json` are written by `tdes/verify.py` from the files on disk.
There are 15 requirements and 85 checks. Where a check could pass vacuously, the verifier also feeds it a broken
copy of real data and requires a FAIL. `tests/test_pipeline.py` runs the whole demo at 12 steps and then attacks
its output 10 ways: an edited ledger record, a re-chained history with a validation span, a repeated batch, a
relabelled lane, a faked token loss, a flipped OPUS decision, a doctored replay, inflated throughput, a corrupted
checkpoint and a missing log event. Each attack turns its requirement to FAIL. Two runs from clean copies of the
repo produced 80 compared files that were identical byte for byte (`tools/compare_runs.py`).

## Limitations

- Most lanes are not A4-admitted. Only agentic and Indic come from A4. The rest are labelled stand-ins.
- The OPUS score is a last-layer heuristic, and its proxy is English text.
- Shares at this scale are coarse. The anneal is 2 steps, and one long row moves a stage by several points.
- The Indic difficulty bands are a length proxy, not a classifier.
- PII regexes are not applied to the stdlib code. They turned digit strings like `"0123456789"` into `<PHONE>`, so
  that source is exempt and marked so in each document entry.
- One process, rank 0. Bit-exact replay is shown on one machine; another torch build may change OPUS scores.
- The committed `artifacts/` include the main and fork checkpoints (most of the repository's size). The reference
  run's checkpoints and token traces, and the pre-crash token traces, are regenerated by `run_demo.py` and ignored.

## Layout

```
run_demo.py                 the one command
config/demo_config.json     demo scale, lane policies, drills (the mixture itself is inputs/a5/mixture.yaml)
inputs/                     vendored A2/A4/A5 artifacts and corpus, PROVENANCE.json
tdes/                       tokenizer adapter, contracts, schedule, corpus, shards, firewall, packing, opus,
                            dataloader, trainer, replay, audit, perf, verify
tests/                      unit tests and the end-to-end tamper tests
tools/                      prepare_inputs.py (dev only, reads the upstream folders), compare_runs.py
artifacts/                  everything run_demo.py generates (committed so it can be inspected without a rerun)
```
