// moe_routing_topk.metal
// GPU-resident MoE routing: softmax + top-K expert selection.
// NOTE: top-K here is a simple serial selection done by thread 0 (correct but
// not the "SG-tournament" the blueprint claims). Optimize later if it matters.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n_experts;
    uint top_k;
    uint norm_topk;
};

kernel void moe_routing_topk(
    device const half*   gate_logits   [[buffer(0)]],
    device int*          topk_indices  [[buffer(1)]],
    device half*         topk_weights  [[buffer(2)]],
    constant Params&     params        [[buffer(3)]],
    uint                 tid           [[thread_position_in_grid]]
) {
    threadgroup float shared_logits[256];

    if (tid < params.n_experts) {
        shared_logits[tid] = (float)gate_logits[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) return;   // serial top-k by thread 0 (n_experts <= 256)

    // softmax (numerically stable)
    float max_val = -INFINITY;
    for (uint i = 0; i < params.n_experts; i++) max_val = max(max_val, shared_logits[i]);
    float sum_exp = 0.0f;
    for (uint i = 0; i < params.n_experts; i++) {
        shared_logits[i] = exp(shared_logits[i] - max_val);
        sum_exp += shared_logits[i];
    }
    for (uint i = 0; i < params.n_experts; i++) shared_logits[i] /= sum_exp;

    // top-k by repeated argmax
    float wsum = 0.0f;
    for (uint k = 0; k < params.top_k; k++) {
        float best = -INFINITY; int bi = -1;
        for (uint i = 0; i < params.n_experts; i++) {
            if (shared_logits[i] > best) { best = shared_logits[i]; bi = (int)i; }
        }
        topk_indices[k] = bi;
        topk_weights[k] = (half)best;
        wsum += best;
        shared_logits[bi] = -INFINITY;
    }
    if (params.norm_topk == 1 && wsum > 0.0f) {
        for (uint k = 0; k < params.top_k; k++)
            topk_weights[k] = (half)((float)topk_weights[k] / wsum);
    }
}
