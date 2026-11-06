#!/usr/bin/env python3
"""Turn a training checkpoint into a model folder for slm.

    python tools/convert.py runs/g86-f/model.pt model --tokenizer data/pretrain/tokenizer.json

Needs torch and safetensors. Accepts model.pt and ckpt.pt files.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--tokenizer", required=True)
    a = p.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.float().clone() for k, v in ck["model"].items() if k != "lm_head.weight"}
    save_file(state, out / "model.safetensors")
    (out / "config.json").write_text(json.dumps(ck["config"], indent=1), encoding="utf-8")
    shutil.copy(a.tokenizer, out / "tokenizer.json")
    print(f"wrote {out}: {sum(v.numel() for v in state.values()) / 1e6:.1f}M parameters")


if __name__ == "__main__":
    main()
