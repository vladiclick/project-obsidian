// fused_residual_rms_norm.metal
// Residual add + RMS norm in one GPU pass.
// Dispatch ONE threadgroup per token, `hidden_dim` threads per group.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n_elements;   // B * L * H
    uint hidden_dim;   // H
    float eps;
};

kernel void fused_residual_rms_norm(
    device const half*   x        [[buffer(0)]],
    device const half*   residual [[buffer(1)]],
    device const half*   weight   [[buffer(2)]],
    device half*         output   [[buffer(3)]],
    constant Params&     params   [[buffer(4)]],
    uint                 tid      [[thread_position_in_grid]],
    uint                 lid      [[thread_index_in_threadgroup]]
) {
    if (tid >= params.n_elements) return;

    float val = (float)x[tid] + (float)residual[tid];

    threadgroup float thread_sq[1024];
    thread_sq[lid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.hidden_dim / 2; stride > 0; stride >>= 1) {
        if (lid < stride) thread_sq[lid] += thread_sq[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = sqrt(thread_sq[0] / (float)params.hidden_dim + params.eps);
    uint h_idx = tid % params.hidden_dim;
    output[tid] = (half)((val / rms) * (float)weight[h_idx]);
}
