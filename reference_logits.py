#!/usr/bin/env python
"""Ground-truth next-token for input id 9707 via the validated AirLLM Python
MoE engine (layer-streamed, fits 16GB, q/k-norm + proper RoPE). Compare its
argmax/top-5 against Obsidian's output."""
import sys
sys.path.insert(0, "/Users/vladi/Desktop/macairllm/airllm/air_llm")
import mlx.core as mx
import mlx.nn as nn
from airllm.airllm_qwen3_moe import AirLLMQwen3Moe

TOK = int(sys.argv[1]) if len(sys.argv) > 1 else 9707
m = AirLLMQwen3Moe("mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit")

def stat(label, a):
    import numpy as np
    v = np.array(a.astype(mx.float32)).ravel()
    print(f"[stat-ref] {label}: min={v.min():.4f} max={v.max():.4f}")

ids = mx.array([[TOK]])
h = m._embed(ids)
stat("embed", h)
caches = [None] * m.n_layers
mask = nn.MultiHeadAttention.create_additive_causal_mask(ids.shape[1]).astype(h.dtype)
for il in range(m.n_layers):
    h, caches[il] = m._layer(h, il, mask, None)
    mx.eval(h, *caches[il])
    if il == 0 or il == 1:
        stat(f"layer{il}", h)
logits = m._head(h[:, -1:])[0, -1]   # [vocab]
mx.eval(logits)
stat("logits", logits)

y = int(mx.argmax(logits).item())
seq = [y]
for _ in range(7):  # 8-token greedy sequence to compare vs Obsidian
    h = m._embed(mx.array([[y]]))
    for il in range(m.n_layers):
        h, caches[il] = m._layer(h, il, None, caches[il])
        mx.eval(h, *caches[il])
    y = int(mx.argmax(m._head(h[:, -1:])[0, -1]).item())
    seq.append(y)
print(f"[airllm] greedy seq from {TOK}: {seq}")
