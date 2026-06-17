// dequant_matvec_8bit.metal — 8-bit dequant + matvec (for the MoE router gate).
// 8-bit values are stored 4-per-uint32; viewed as bytes, value i is byte i.
#include <metal_stdlib>
using namespace metal;

struct Params { uint in_features; uint out_features; uint group_size; };

kernel void dequant_matvec_8bit(
    device const uchar*  w_q     [[buffer(0)]],
    device const half*   scales  [[buffer(1)]],
    device const half*   biases  [[buffer(2)]],
    device const half*   input   [[buffer(3)]],
    device half*         output  [[buffer(4)]],
    constant Params&     p       [[buffer(5)]],
    uint                 tid     [[thread_position_in_grid]]
) {
    if (tid >= p.out_features) return;
    uint groups = p.in_features / p.group_size;
    float acc = 0.0f;
    uint row = tid * p.in_features;
    for (uint g = 0; g < groups; g++) {
        float sc = (float)scales[tid * groups + g];
        float bi = (float)biases[tid * groups + g];
        for (uint j = 0; j < p.group_size; j++) {
            uint col = g * p.group_size + j;
            float byte = (float)w_q[row + col];
            acc += (byte * sc + bi) * (float)input[col];
        }
    }
    output[tid] = (half)acc;
}
