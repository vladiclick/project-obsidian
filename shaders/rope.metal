// rope.metal — Rotary Positional Embedding (NeoX/HF style, half-split).
// In-place on [n_heads, head_dim] for a single position (decode). theta = config
// rope_theta (Qwen3-30B-A3B = 1e7). One thread per (head, dim-pair).

#include <metal_stdlib>
using namespace metal;

struct RopeParams { uint n_heads; uint head_dim; uint pos; float theta; };

kernel void rope_apply(
    device half*         x   [[buffer(0)]],   // [n_heads, head_dim], modified in place
    constant RopeParams& p   [[buffer(1)]],
    uint                 tid [[thread_position_in_grid]]
) {
    uint half_d = p.head_dim / 2;
    if (tid >= p.n_heads * half_d) return;
    uint h = tid / half_d;
    uint i = tid % half_d;
    float freq  = pow(p.theta, -2.0f * (float)i / (float)p.head_dim);
    float angle = (float)p.pos * freq;
    float c = cos(angle), s = sin(angle);
    uint base = h * p.head_dim;
    float x0 = (float)x[base + i];
    float x1 = (float)x[base + i + half_d];
    x[base + i]          = (half)(x0 * c - x1 * s);
    x[base + i + half_d] = (half)(x1 * c + x0 * s);
}
