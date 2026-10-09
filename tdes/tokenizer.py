"""Adapter that executes the frozen A2 tokenizer contract inside the data system.

The text -> id mapping is exactly A2's `tokenizer.json` (10,000-token BPE for en/hi/te/ur: NFC, every run of
whitespace collapsed to one space, Metaspace pre-tokenisation, BPE, `<unk>` = id 0, no byte fallback). The file
is vendored byte-for-byte and its SHA-256 must equal the hash A2 published (pinned in config).

A2 ships one special token (`<unk>`) and no EOS, PAD or role markers, but packing needs all of them. The adapter
appends a fixed, versioned block of control tokens *after* the A2 vocabulary (ids 10000+). They are never
produced from text: `encode()` is A2's encoder, and the control ids are inserted only by `render_document()`.
A user string such as "<|eos|>" therefore tokenizes as ordinary text and cannot forge a boundary.

Identity. `tokenizer_id` = sha256(canonical {A2 file sha256, control-token block, rendering rules}). Shards and
ledger records carry this id; changing the A2 file, the control block or the rendering rules changes the id, so
an old shard cannot silently stay valid under a new tokenizer.
"""
import json
from pathlib import Path

from .hashing import canonical_json, sha256_bytes, sha256_file

CONTROL_TOKENS = ["<|pad|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|tool_call|>", "<|tool|>"]
ADAPTER_VERSION = "tdes-a2-adapter/1"
# roles whose tokens are loss-bearing targets in structured (SFT / agentic) samples
TRAINABLE_ROLES = {"assistant", "tool_call"}
RENDERING_RULES = {
    "plain": "encode(text) + [<|eos|>]; every token and the EOS are loss-eligible",
    "messages": "per message <|role|> + encode(content); an assistant tool call is <|tool_call|> + "
                "encode(canonical JSON of the call); only assistant and tool_call tokens are loss-eligible; "
                "a final <|eos|> is loss-eligible",
}


class TokenizerHashMismatch(Exception):
    pass


class A2Tokenizer:
    def __init__(self, path, expected_a2_sha256=None, expected_tokenizer_id=None):
        from tokenizers import Tokenizer
        self.path = Path(path)
        self.a2_sha256 = sha256_file(self.path)
        if expected_a2_sha256 is not None and self.a2_sha256 != expected_a2_sha256:
            raise TokenizerHashMismatch(f"A2 tokenizer file sha256 {self.a2_sha256} != pinned {expected_a2_sha256}")
        self._tok = Tokenizer.from_file(str(self.path))
        self.base_vocab = self._tok.get_vocab_size()
        added = json.loads(self.path.read_text(encoding="utf-8"))["added_tokens"]
        if added:
            raise TokenizerHashMismatch("A2 tokenizer unexpectedly defines added tokens")
        self.unk_id = self._tok.token_to_id("<unk>")
        self.special_ids = {s: self.base_vocab + i for i, s in enumerate(CONTROL_TOKENS)}
        self.vocab_size = self.base_vocab + len(CONTROL_TOKENS)
        self.pad_id = self.special_ids["<|pad|>"]
        self.eos_id = self.special_ids["<|eos|>"]
        self.spec = {"adapter": ADAPTER_VERSION, "a2_tokenizer_sha256": self.a2_sha256, "a2_vocab_size": self.base_vocab,
                     "control_tokens": self.special_ids, "rendering": RENDERING_RULES,
                     "trainable_roles": sorted(TRAINABLE_ROLES)}
        self.tokenizer_id = sha256_bytes(canonical_json(self.spec))
        if expected_tokenizer_id is not None and self.tokenizer_id != expected_tokenizer_id:
            raise TokenizerHashMismatch(f"tokenizer id {self.tokenizer_id} != pinned {expected_tokenizer_id}")
        self._inv = {v: k for k, v in self.special_ids.items()}

    # ---- A2 text mapping -----------------------------------------------------------------
    def encode(self, text: str):
        return self._tok.encode(text).ids

    def normalize(self, text: str) -> str:
        return self._tok.normalizer.normalize_str(text)

    def decode(self, ids) -> str:
        out, buf = [], []
        for t in ids:
            t = int(t)
            if t in self._inv:
                if buf:
                    out.append(self._tok.decode(buf))
                    buf = []
                out.append(self._inv[t])
            else:
                buf.append(t)
        if buf:
            out.append(self._tok.decode(buf))
        return " ".join(out)

    def token_preview(self, t: int) -> str:
        t = int(t)
        return self._inv.get(t) or self._tok.id_to_token(t)

    def is_special(self, t: int) -> bool:
        return int(t) >= self.base_vocab

    def unk_rate(self, text: str) -> float:
        ids = self.encode(text)
        return ids.count(self.unk_id) / max(1, len(ids))

    # ---- documents ---------------------------------------------------------------------------
    def render_document(self, doc):
        """Token ids + per-token loss eligibility for one admitted document, EOS appended."""
        if "messages" in doc:
            ids, elig = [], []
            for m in doc["messages"]:
                if m.get("tool_call"):
                    role, content = "tool_call", json.dumps(m["tool_call"], sort_keys=True, ensure_ascii=False)
                else:
                    role, content = m["role"], m["content"] or ""
                body = [self.special_ids[f"<|{role}|>"]] + self.encode(content)
                ids.extend(body)
                elig.extend([1 if role in TRAINABLE_ROLES else 0] * len(body))
            ids.append(self.eos_id)
            elig.append(1)
            return ids, elig
        ids = self.encode(doc["text"]) + [self.eos_id]
        return ids, [1] * len(ids)
