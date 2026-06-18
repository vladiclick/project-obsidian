// matmul_dequant_4bit.metal — BATCHED 4-bit dequant matmul (MEGAPROMPT V3).
// Y[T, out] = X[T, in] @ dequant(W[out, in]).
//
// Mirrors fused_dequant_matvec_4bit's dequant math EXACTLY (same group layout,
// same `acc += fma(w, x, 0.0f)` accumulation order) so a single output is
// bit-identical to the per-token matvec — but one thread per (token, output)
// means a batch of T tokens runs in ONE dispatch with T× the threads, taking
// GPU occupancy from <5% (matvec) toward saturation. Weights stay 4-bit packed
// (no f16 weight matrix exists in this engine).

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint in_features;
    uint out_features;
    uint group_size;
    uint n_tokens;
};

kernel void matmul_dequant_4bit(
    device const uchar*  w_q      [[buffer(0)]],   // [out, in/2] packed nibbles
    device const half*   scales   [[buffer(1)]],   // [out, in/group]
    device const half*   biases   [[buffer(2)]],   // [out, in/group]
    device const half*   input    [[buffer(3)]],   // [T, in]
    device half*         output   [[buffer(4)]],   // [T, out]
    constant Params&     p        [[buffer(5)]],
    uint                 tid      [[thread_position_in_grid]]
) {
    uint total = p.n_tokens * p.out_features;
    if (tid >= total) return;

    uint tok = tid / p.out_features;   // which token row
    uint o   = tid % p.out_features;   // which output feature

    uint in_features = p.in_features;
    uint group_size  = p.group_size;
    uint groups      = in_features / group_size;

    device const half* x = input + (uint)tok * in_features;

    float acc = 0.0f;
    uint row_offset = o * in_features / 2;

    for (uint g = 0; g < groups; g++) {
        uint scale_idx = o * groups + g;
        float scale = (float)scales[scale_idx];
        float bias  = (float)biases[scale_idx];

        for (uint j = 0; j < group_size; j += 2) {
            uint col = g * group_size + j;
            uchar packed = w_q[row_offset + col / 2];

            uint n0 = packed & 0x0F;
            uint n1 = (packed >> 4) & 0x0F;

            if (col < in_features) {
                acc += fma((float)n0 * scale + bias, (float)x[col], 0.0f);
            }
            if (col + 1 < in_features) {
                acc += fma((float)n1 * scale + bias, (float)x[col + 1], 0.0f);
            }
        }
    }
    output[(uint)tok * p.out_features + o] = (half)acc;
}
