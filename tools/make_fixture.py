#!/usr/bin/env python3
"""Save reference logits from the PyTorch model for `slm verify`.

    python tools/make_fixture.py --lr-root <folder containing lr/> tiny tests/fixtures/tiny
    python tools/make_fixture.py --lr-root <...> real model --checkpoint model.pt --val val.bin

`tiny` builds a small model with random weights and is what CI checks every backend
against. `real` adds expected.safetensors to a converted model folder.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def expected(model, tokens, loops, positions):
    """Reference logits from step(), the same path slm decodes with."""
    state = model.init_state(1, loops, "cpu", torch.float32)
    keep = []
    with torch.no_grad():
        for t, tok in enumerate(tokens):
            logits, state = model.step(torch.tensor([[int(tok)]]), torch.tensor([float(t)]), state, loops)
            if t in positions:
                keep.append(logits[0, 0].clone())
    return {
        "tokens": torch.tensor([float(t) for t in tokens], dtype=torch.float32),
        "loops": torch.tensor([float(loops)], dtype=torch.float32),
        "positions": torch.tensor([float(p) for p in positions], dtype=torch.float32),
        "logits": torch.stack(keep),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr-root", required=True)
    p.add_argument("mode", choices=["tiny", "real"])
    p.add_argument("out")
    p.add_argument("--checkpoint")
    p.add_argument("--val")
    p.add_argument("--tokens", type=int, default=6000)
    p.add_argument("--loops", type=int, default=4)
    a = p.parse_args()
    sys.path.insert(0, a.lr_root)
    from lr import LoopedHybridReasoner
    from lr.config import ModelConfig

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "tiny":
        torch.manual_seed(0)
        cfg = ModelConfig(vocab_size=97, d_model=32, n_heads=4, global_kv_heads=2, global_logn=16,
                          global_kshift=1, global_emb=1, global_far=1, prelude="S", core="LS", coda="SG",
                          window=8, max_loops=6, default_loops=3)
        model = LoopedHybridReasoner(cfg).eval()
        state = {k: v.clone() for k, v in model.state_dict().items() if k != "lm_head.weight"}
        save_file(state, out / "model.safetensors")
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1), encoding="utf-8")
        tokens = torch.randint(0, cfg.vocab_size, (40,)).tolist()
        positions = list(range(40))
        loops = 3
    else:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        model = LoopedHybridReasoner(ModelConfig(**ck["config"])).eval()
        model.load_state_dict(ck["model"])
        tokens = np.fromfile(a.val, dtype=np.uint16, count=a.tokens).astype(int).tolist()
        positions = [0, 1, 100, 4095, 4096, 4097, len(tokens) - 2, len(tokens) - 1]
        loops = a.loops
    save_file(expected(model, tokens, loops, positions), out / "expected.safetensors")


if __name__ == "__main__":
    main()
