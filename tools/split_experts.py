#!/usr/bin/env python3
"""Split a Qwen3-MoE 4-bit MLX checkpoint's stacked `switch_mlp` experts into
per-layer, per-expert safetensors for the Obsidian Rust engine.

Source tensors (stacked on axis 0 = expert):
    model.layers.{il}.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}

Output (one file per layer), matching disk_io.rs ExpertIndex
(name = "e{N}.{proj}.{kind}", exactly 3 dot-separated parts):
    <model_dir>/splitted_moe/layer_{il}_experts.safetensors
      e{e}.gate_proj.weight  (uint32, packed 4-bit)
      e{e}.gate_proj.scales  (float16)   e{e}.gate_proj.biases (float16)
      e{e}.up_proj.{weight,scales,biases}
      e{e}.down_proj.{weight,scales,biases}

Weight stays uint32 (packed); scales/biases cast bf16 -> float16 (Obsidian
dequant kernels expect 2-byte f16). Router gate (mlp.gate, 8-bit) is NOT split
— it's a non-expert tensor loaded by weight_loader.rs.
"""
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx

PROJS = ["gate_proj", "up_proj", "down_proj"]
KINDS = ["weight", "scales", "biases"]
KEY_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.switch_mlp\.(\w+)\.(\w+)")


def main():
    model_dir = Path(sys.argv[1]).resolve()
    out_dir = model_dir / "splitted_moe"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((model_dir / "config.json").read_text())
    n_layers = cfg["num_hidden_layers"]
    n_exp = cfg["num_experts"]
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]

    print(f"[split] model={model_dir.name} layers={n_layers} experts={n_exp}", flush=True)
    print(f"[split] out={out_dir}", flush=True)

    t0 = time.time()
    done = 0
    for il in range(n_layers):
        target = out_dir / f"layer_{il}_experts.safetensors"
        if target.exists():
            # idempotent: skip layers already produced (allows resume)
            print(f"[split] layer {il}: exists, skip", flush=True)
            done += 1
            continue

        # group this layer's 9 tensors by their shard so each shard loads once
        shards = {}
        for p in PROJS:
            for k in KINDS:
                key = f"model.layers.{il}.mlp.switch_mlp.{p}.{k}"
                shards.setdefault(idx[key], []).append((p, k, key))

        out = {}
        for shard, items in shards.items():
            w = mx.load(str(model_dir / shard))  # lazy mmap; OS page-caches re-loads
            for p, k, key in items:
                t = w[key]
                if t.shape[0] != n_exp:
                    raise SystemExit(
                        f"FATAL {key}: dim0={t.shape[0]} != num_experts={n_exp} "
                        f"(expert axis assumption wrong)")
                if k in ("scales", "biases"):
                    t = t.astype(mx.float16)
                for e in range(n_exp):
                    out[f"e{e}.{p}.{k}"] = t[e]
            mx.eval(list(out.values()))
            del w

        mx.save_safetensors(str(target), out)
        del out
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        done += 1
        dt = time.time() - t0
        print(f"[split] layer {il}: {n_exp*9} tensors -> {target.name} "
              f"({done}/{n_layers}, {dt:.0f}s, ~{dt/done:.1f}s/layer)", flush=True)

    # ---- self-check ----
    from safetensors import safe_open
    for il in (0, n_layers - 1):
        f = out_dir / f"layer_{il}_experts.safetensors"
        with safe_open(str(f), framework="numpy") as sf:
            names = list(sf.keys())
        assert len(names) == n_exp * 9, f"layer {il}: {len(names)} != {n_exp*9}"
        for nm in names[:3]:
            parts = nm.split(".")
            assert len(parts) == 3 and parts[0][0] == "e", f"bad name {nm}"
    print(f"[split] DONE: {n_layers} layer files in {time.time()-t0:.0f}s, self-check OK", flush=True)


if __name__ == "__main__":
    main()
