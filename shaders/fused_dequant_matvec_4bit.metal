// fused_dequant_matvec_4bit.metal
// FMA-optimized 4-bit dequant + matrix-vector multiply.
// Thread grid: one thread per output row (out_features threads total).

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint in_features;
    uint out_features;
    uint group_size;
};

kernel void fused_dequant_matvec_4bit(
    device const uchar*  w_q        [[buffer(0)]],
    device const half*   scales     [[buffer(1)]],
    device const half*   biases     [[buffer(2)]],
    device const half*   input      [[buffer(3)]],
    device half*         output     [[buffer(4)]],
    constant Params&     params     [[buffer(5)]],
    uint                 tid        [[thread_position_in_grid]]
) {
    if (tid >= params.out_features) return;

    uint in_features = params.in_features;
    uint group_size  = params.group_size;
    uint groups      = in_features / group_size;

    float acc = 0.0f;
    uint row_offset = tid * in_features / 2;  // 2 nibbles per byte

    for (uint g = 0; g < groups; g++) {
        uint scale_idx = tid * groups + g;
        float scale = (float)scales[scale_idx];
        float bias  = (float)biases[scale_idx];

        for (uint j = 0; j < group_size; j += 2) {
            uint col = g * group_size + j;
            uchar packed = w_q[row_offset + col / 2];

            uint n0 = packed & 0x0F;
            uint n1 = (packed >> 4) & 0x0F;

            if (col < in_features) {
                acc += fma((float)n0 * scale + bias, (float)input[col], 0.0f);
            }
            if (col + 1 < in_features) {
                acc += fma((float)n1 * scale + bias, (float)input[col + 1], 0.0f);
            }
        }
    }
    output[tid] = (half)acc;
}
