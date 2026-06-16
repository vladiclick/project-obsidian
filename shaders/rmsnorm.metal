// rmsnorm.metal — plain RMSNorm over the last `dim` elements.
// Dispatch one threadgroup per row, FIXED 256 threads (Metal max/tg is ~1024;
// dim can be 2048+, so each thread strides over multiple elements).
#include <metal_stdlib>
using namespace metal;

struct RmsParams { uint dim; float eps; };

kernel void rmsnorm(
    device const half*  x      [[buffer(0)]],   // [n_rows, dim]
    device const half*  weight [[buffer(1)]],   // [dim]
    device half*        out    [[buffer(2)]],   // [n_rows, dim]
    constant RmsParams& p      [[buffer(3)]],
    uint                row    [[threadgroup_position_in_grid]],
    uint                lid    [[thread_position_in_threadgroup]]
) {
    threadgroup float sq[256];
    uint base = row * p.dim;

    float ps = 0.0f;
    for (uint i = lid; i < p.dim; i += 256) {
        float v = (float)x[base + i];
        ps += v * v;
    }
    sq[lid] = ps;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s) sq[lid] += sq[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(sq[0] / (float)p.dim + p.eps);

    for (uint i = lid; i < p.dim; i += 256) {
        out[base + i] = (half)(((float)x[base + i] / rms) * (float)weight[i]);
    }
}
