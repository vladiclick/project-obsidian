// fused_moe_combine.metal
// Weighted sum of expert SwiGLU outputs + residual add, in one kernel.
// NOTE: blueprint folded RMS norm in here too but its single-pass RMS was
// incorrect (no reduction). RMS norm is left to fused_residual_rms_norm.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint hidden_dim;
    uint top_k;
    float eps;
};

kernel void fused_moe_combine(
    device const half*   expert_gate     [[buffer(0)]],  // [K, H]
    device const half*   expert_up       [[buffer(1)]],  // [K, H]
    device const half*   routing_weights [[buffer(3)]],  // [K]
    device const half*   residual        [[buffer(4)]],  // [H]
    device half*         output          [[buffer(5)]],  // [H]
    constant Params&     params          [[buffer(7)]],
    uint                 tid             [[thread_position_in_grid]]
) {
    if (tid >= params.hidden_dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < params.top_k; k++) {
        float rw = (float)routing_weights[k];
        float g  = (float)expert_gate[k * params.hidden_dim + tid];
        float u  = (float)expert_up[k * params.hidden_dim + tid];
        float silu_g = g / (1.0f + exp(-g));   // SwiGLU
        acc += rw * silu_g * u;
    }
    acc += (float)residual[tid];               // residual add
    output[tid] = (half)acc;
}
