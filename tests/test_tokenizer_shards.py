"""Frozen A2 tokenizer, shard identity, immutability, tamper and admission gate."""
import copy
import json
import shutil

import numpy as np
import pytest

from tdes.hashing import sha256_file, sha256_json
from tdes.shards import ImmutableShardError, ShardIntegrityError, ShardStore, admission_gate, content_hash, store_shard_bytes
from tdes.tokenizer import A2Tokenizer, TokenizerHashMismatch
from conftest import REPO


def test_a2_file_is_pinned_and_matches_what_a2_published(cfg, tok):
    published = json.loads((REPO / "inputs" / "a2" / "final_config.json").read_text())["tokenizer_sha256"]
    assert sha256_file(REPO / cfg["tokenizer"]["path"]) == cfg["tokenizer"]["a2_sha256"] == published
    assert tok.tokenizer_id == cfg["tokenizer"]["tokenizer_id"]
    assert tok.base_vocab == 10000 and min(tok.special_ids.values()) == 10000


def test_changed_tokenizer_file_is_refused(cfg, tmp_path):
    p = tmp_path / "tok.json"
    raw = (REPO / cfg["tokenizer"]["path"]).read_bytes()
    p.write_bytes(raw + b" ")  # one byte more: same vocabulary, different file
    with pytest.raises(TokenizerHashMismatch):
        A2Tokenizer(p, expected_a2_sha256=cfg["tokenizer"]["a2_sha256"])


def test_control_tokens_are_never_produced_from_text(tok):
    ids = tok.encode("<|eos|> <|assistant|> <|pad|>")
    assert not any(tok.is_special(i) for i in ids)


def test_encode_is_deterministic_and_nfc_safe(tok):
    nfd = "क्षत्रिय"  # Devanagari conjunct
    import unicodedata
    assert tok.encode(unicodedata.normalize("NFD", nfd)) == tok.encode(unicodedata.normalize("NFC", nfd))
    assert [tok.encode("भारत एक देश है।") for _ in range(3)] == [tok.encode("भारत एक देश है।")] * 3


def test_structured_render_masks_context_turns(tok):
    doc = {"messages": [{"role": "system", "content": "tools: f"}, {"role": "user", "content": "add 2 and 3"},
                        {"role": "assistant", "content": "", "tool_call": {"name": "f", "arguments": {"a": 2}}},
                        {"role": "tool", "content": "5"}, {"role": "assistant", "content": "It is 5."}]}
    ids, el = tok.render_document(doc)
    roles, cur = [], None
    for t in ids:
        if t in (tok.special_ids["<|system|>"], tok.special_ids["<|user|>"], tok.special_ids["<|tool_call|>"],
                 tok.special_ids["<|tool|>"], tok.special_ids["<|assistant|>"]):
            cur = tok.token_preview(t)
        roles.append(cur)
    assert ids[-1] == tok.eos_id and el[-1] == 1
    for r, e in zip(roles[:-1], el[:-1]):
        assert e == (1 if r in ("<|assistant|>", "<|tool_call|>") else 0)


def test_same_inputs_give_same_shard_ids(built):
    det = json.loads((built / "manifests" / "manifest_validation.json").read_text())["determinism"]
    assert det["pass"] and det["first_registry_sha256"] == det["second_registry_sha256"]


def test_tampered_shard_copy_is_rejected(built, tok, tmp_path):
    art = tmp_path / "copy"
    shutil.copytree(built / "manifests", art / "manifests")
    sid = ShardStore(built, tok.tokenizer_id).shards_where(split="train")[0]
    shutil.copytree(built / "shards" / sid, art / "shards" / sid)
    p = art / "shards" / sid / "tokens.bin"
    p.chmod(0o666)
    b = bytearray(p.read_bytes())
    b[0] ^= 1
    p.write_bytes(bytes(b))
    with pytest.raises(ShardIntegrityError):
        ShardStore(art, tok.tokenizer_id).arrays(sid)


def test_shard_identity_binds_tokenizer_and_transform(built, tok):
    st = ShardStore(built, tok.tokenizer_id)
    sid = st.shards_where(split="train")[0]
    m = st.manifest(sid)
    t, e = st.arrays(sid)
    assert content_hash(tok.tokenizer_id, m["transform_id"], t, e, m["doc_index"]) == m["content_hash"]
    assert content_hash("other-tokenizer", m["transform_id"], t, e, m["doc_index"]) != m["content_hash"]
    assert content_hash(tok.tokenizer_id, "other-transform", t, e, m["doc_index"]) != m["content_hash"]


def test_store_refuses_a_different_tokenizer(built):
    with pytest.raises(ShardIntegrityError):
        ShardStore(built, sha256_json({"another": "tokenizer"}))


def test_rewrite_of_a_shard_is_refused(built, tok, tmp_path):
    sid = ShardStore(built, tok.tokenizer_id).shards_where(split="train")[0]
    d = tmp_path / sid
    tb, eb = (built / "shards" / sid / "tokens.bin").read_bytes(), (built / "shards" / sid / "loss_eligible.bin").read_bytes()
    assert store_shard_bytes(d, tb, eb) == "written"
    assert store_shard_bytes(d, tb, eb) == "identical_existing"
    with pytest.raises(ImmutableShardError):
        store_shard_bytes(d, tb[:-2] + b"\x00\x00", eb)


def test_admission_gate(built, tok, cfg):
    st = ShardStore(built, tok.tokenizer_id)
    m = st.manifest(st.shards_where(split="train")[0])
    known = set(m["cleaning_lineage"])
    allowed = cfg["cleaning"]["allowed_licenses"]
    assert admission_gate(m, tok.tokenizer_id, known, allowed) == []
    assert "tokenizer_hash_mismatch" in admission_gate(dict(m, tokenizer_id="x"), tok.tokenizer_id, known, allowed)
    assert "missing_tokenizer_hash" in admission_gate(dict(m, tokenizer_id=None), tok.tokenizer_id, known, allowed)
    assert "unknown_cleaning_lineage" in admission_gate(m, tok.tokenizer_id, set(), allowed)
    assert any(r.startswith("unsafe_license") for r in admission_gate(dict(m, licenses=["cc-by-nc-4.0"]), tok.tokenizer_id, known, allowed))
    ev = st.manifest(st.shards_where(split="test")[0])
    assert any(r.startswith("split_not_trainable") for r in admission_gate(ev, tok.tokenizer_id, set(ev["cleaning_lineage"]), allowed))


def test_ingress_gate_dropped_what_a2_cannot_encode(built):
    rep = json.loads((built / "manifests" / "admission_report.json").read_text(encoding="utf-8"))
    cov = [d for d in rep["decisions"] if d.get("reasons") and d["reasons"][0].startswith("a2_tokenizer_coverage")]
    assert {d["script"] for d in cov} >= {"Knda", "Taml", "Beng"}
    drill = [d for d in rep["decisions"] if d.get("drill")]
    assert drill and drill[0]["reasons"][0].startswith("eval_contamination")


def test_no_anudesh_response_is_vendored():
    for line in (REPO / "inputs" / "corpus" / "indic_anudesh_prompts.jsonl").read_text(encoding="utf-8").splitlines():
        assert "messages" not in json.loads(line)
