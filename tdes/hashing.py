"""Canonical serialisation and hashing helpers used everywhere.

All identities in the system are SHA-256 over canonical bytes: JSON with sorted
keys and no whitespace, or numpy arrays in a fixed dtype and little-endian
order. Nothing depends on Python's per-process hash randomisation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(canonical_json(obj))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text_file(path) -> str:
    """Hash of a text file with line endings normalised to LF (stable across git checkouts)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return sha256_bytes(data)


def array_bytes(a, dtype) -> bytes:
    arr = np.ascontiguousarray(np.asarray(a, dtype=dtype))
    return arr.astype(np.dtype(dtype).newbyteorder("<"), copy=False).tobytes()


def sha256_arrays(*pairs) -> str:
    """Hash a sequence of (array, dtype) pairs, with each array's shape included."""
    h = hashlib.sha256()
    for a, dt in pairs:
        arr = np.asarray(a)
        h.update(canonical_json(list(arr.shape)))
        h.update(array_bytes(arr, dt))
    return h.hexdigest()


def stable_u64(*parts) -> int:
    """Deterministic 64-bit integer from arbitrary string parts (used for shuffles)."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")
