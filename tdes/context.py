"""Loads what every process needs from disk: config, frozen A2 tokenizer, A5 plan, shard store, schedule."""
import json
from pathlib import Path

import torch

from .contracts import load_a5_plan
from .shards import ShardStore
from .tokenizer import A2Tokenizer

REPO = Path(__file__).resolve().parents[1]


def load_config(path=None):
    return json.loads(Path(path or REPO / "config" / "demo_config.json").read_text(encoding="utf-8"))


def load_tokenizer(cfg):
    t = cfg["tokenizer"]
    return A2Tokenizer(REPO / t["path"], expected_a2_sha256=t["a2_sha256"], expected_tokenizer_id=t["tokenizer_id"])


def load_schedule(artifacts, name="mixture_schedule.json"):
    return json.loads((Path(artifacts) / "manifests" / name).read_text(encoding="utf-8"))


def setup_determinism():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)


class Ctx:
    def __init__(self, artifacts, cfg_path=None):
        self.artifacts = Path(artifacts)
        self.cfg = load_config(cfg_path)
        self.tok = load_tokenizer(self.cfg)
        self.plan = load_a5_plan(self.cfg, REPO)
        self.store = ShardStore(self.artifacts, self.tok.tokenizer_id)
