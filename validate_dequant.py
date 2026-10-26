#!/usr/bin/env python
"""Ground-truth check for Obsidian's 4-bit dequant packing.
Compares a numpy replica of the Metal fused_dequant_matvec_4bit unpacking
against MLX's own mx.dequantize on a REAL Qwen3 weight. No full-model load.
"""
import glob, json, os, sys
import mlx.core as mx
import numpy as np

HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
SNAP = glob.glob(f"{HUB}/models--mlx-community--Qwen3-30B-A3B-Instruct-2507-4bit/snapshots/*")[0]
KEY = "model.layers.0.self_attn.q_proj.weight"

idx = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
shard = idx[KEY]
t = mx.load(f"{SNAP}/{shard}")  # lazy; only accessed tensors materialize
W, S, B = t[KEY], t[KEY.replace(".weight", ".scales")], t[KEY.replace(".weight", ".biases")]
bits, group = 4, 64

# MLX ground truth
ref = np.array(mx.dequantize(W, S, B, group_size=group, bits=bits).astype(mx.float32))
out_f, in_f = ref.shape

# Replica of the Metal shader: nibble j of uint32 word = (word >> 4j) & 0xF;
# element i lives at word i//8, nibble i%8; value = nibble*scale[g] + bias[g].
Wn = np.array(W).astype(np.uint32)                       # [out, in/8]
Sn = np.array(S.astype(mx.float32))
Bn = np.array(B.astype(mx.float32))
nib = np.stack([(Wn >> (4 * j)) & 0xF for j in range(8)], -1).reshape(out_f, in_f).astype(np.float32)
mine = nib * np.repeat(Sn, group, axis=1) + np.repeat(Bn, group, axis=1)

diff = np.abs(ref - mine).max()
print(f"[validate] {KEY}  shape={ref.shape}")
print(f"[validate] max abs diff (mine vs mx.dequantize) = {diff:.6g}")
print("Dequant Match: SUCCESS" if diff < 1e-2 else "Dequant Match: FAIL")
sys.exit(0 if diff < 1e-2 else 1)
