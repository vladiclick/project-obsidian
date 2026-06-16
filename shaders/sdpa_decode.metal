// sdpa_decode.metal — single-query GQA attention over the KV cache.
// One threadgroup per query head; threads split the score computation over the
// sequence, then thread 0 does softmax + weighted value sum. seq_len <= 4096.

#include <metal_stdlib>
using namespace metal;

struct SdpaParams { uint n_heads; uint n_kv; uint head_dim; uint seq_len; float scale; };

kernel void sdpa_decode(
    device const half*   q   [[buffer(0)]],   // [n_heads, head_dim]
    device const half*   k   [[buffer(1)]],   // [n_kv, seq_len, head_dim]
    device const half*   v   [[buffer(2)]],   // [n_kv, seq_len, head_dim]
    device half*         out [[buffer(3)]],   // [n_heads, head_dim]
    constant SdpaParams& p   [[buffer(4)]],
    uint                 h   [[threadgroup_position_in_grid]],
    uint                 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float scores[4096];
    uint kv = h / (p.n_heads / p.n_kv);     // GQA: map query head -> kv head

    for (uint t = lid; t < p.seq_len; t += 256) {
        float dot = 0.0f;
        for (uint d = 0; d < p.head_dim; d++)
            dot += (float)q[h * p.head_dim + d]
                 * (float)k[(kv * p.seq_len + t) * p.head_dim + d];
        scores[t] = dot * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float mx = -INFINITY;
        for (uint t = 0; t < p.seq_len; t++) mx = max(mx, scores[t]);
        float sum = 0.0f;
        for (uint t = 0; t < p.seq_len; t++) { scores[t] = exp(scores[t] - mx); sum += scores[t]; }
        for (uint d = 0; d < p.head_dim; d++) {
            float acc = 0.0f;
            for (uint t = 0; t < p.seq_len; t++)
                acc += scores[t] * (float)v[(kv * p.seq_len + t) * p.head_dim + d];
            out[h * p.head_dim + d] = (half)(acc / sum);
        }
    }
}
